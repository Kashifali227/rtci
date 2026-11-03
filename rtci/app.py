# SPDX-License-Identifier: GPL-3.0-only
# rtci, 2026 Kashif Ali.
"""
Real-time cochlear-implant vocoder simulator for the Advanced Bionics HiRes Fidelity 120 strategy.

The headset microphone feeds a streaming, GPyT-exact F120 strategy. Its electrodogram is rendered by a vocoder
and played to the headset.

    rtci                                         GUI: devices, CI/natural A/B, switches, live electrodogram
    rtci --list                                  list audio devices
    rtci --nogui --config ssd --ci-side left     terminal mode (keys: c n m v b u d k g a r s f + - q)
    rtci --file in.wav --outfile out.wav         offline render through the same real-time path
    rtci --file in.wav --config ssd --ci-side right --vocoder get --outfile out.wav

`python -m rtci` is equivalent to `rtci`. In the GUI, the space bar switches between CI and natural sound for
quick A/B listening, and V cycles through the vocoders (GPyT, noise-band, sine, GET). Offline renders are stereo
[left, right] in the chosen hearing configuration. A mono input file feeds both microphones.
"""
from __future__ import annotations
from . import _svml  # noqa: F401,I001  (import before numba to enable SVML vector math; see _svml.py)

import argparse
import os
import sys
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np

os.environ.setdefault("SD_ENABLE_ASIO", "1")      # expose installed ASIO drivers, such as Yamaha Steinberg USB ASIO

HERE = Path(__file__).resolve().parent
E = None                     # rtci.engine, imported in main(): it needs GPyT, which --help and --list must not require


def _load_engine():
    global E
    if E is None:
        from . import engine
        E = engine
    return E


FS = 48000


# ------------------------------------------------------------------------------------------------ devices
def list_devices(api_filter=None):
    import sounddevice as sd
    apis = sd.query_hostapis()
    out = []
    for d in sd.query_devices():
        api = apis[d["hostapi"]]["name"]
        if api_filter and api_filter.lower() not in api.lower():
            continue
        out.append(dict(index=d["index"], name=d["name"].split("\n")[0], api=api,
                        nin=d["max_input_channels"], nout=d["max_output_channels"], fs=d["default_samplerate"]))
    return out


PREFERRED = ("steinberg", "yamaha", "ur22", "ur12", "ur44", "ur24", "ur-c")


def default_hostapi():
    """Return the default host API for this platform.

    On Windows this is ASIO if a Steinberg/Yamaha ASIO driver is installed, else WASAPI. Measured on a UR22mkII
    at 44.1 kHz with 128-sample blocks, including the CI chain, ASIO gives 57 ms in total at the driver's default
    512-sample buffer and WASAPI 69-71 ms through the same Yamaha driver (58 ms with the generic class driver).
    A smaller buffer in the Yamaha Steinberg USB Driver control panel (ASIO tab, Buffer Size 64 or 128) lowers
    the ASIO figure further. macOS uses Core Audio and Linux uses ALSA; JACK, if available, can be chosen with
    --hostapi JACK."""
    if sys.platform == "darwin":
        return "Core Audio"
    if not sys.platform.startswith("win"):
        return "ALSA"
    for d in list_devices("ASIO"):
        if any(k in d["name"].lower() for k in PREFERRED):
            return "ASIO"
    return "Windows WASAPI"


def find_device(spec, kind, api="WASAPI"):
    """spec is a device index, a name substring, or None. None picks a Steinberg/Yamaha interface if one is
    present, else the host API's default device."""
    import sounddevice as sd
    if spec is None:
        key = "nin" if kind == "input" else "nout"
        for d in list_devices(api):
            if d[key] > 0 and any(k in d["name"].lower() for k in PREFERRED):
                return d["index"]
        for a in sd.query_hostapis():
            if api.lower() in a["name"].lower():
                return a["default_input_device" if kind == "input" else "default_output_device"]
        return None
    if str(spec).isdigit():
        return int(spec)
    key = "nin" if kind == "input" else "nout"
    for d in list_devices(api) + list_devices():
        if d[key] > 0 and str(spec).lower() in d["name"].lower():
            return d["index"]
    raise SystemExit(f"no {kind} device matching '{spec}' (try --list)")


# ------------------------------------------------------------------------------------------------ offline
def render_file(src, dst, eng, trim=True):
    """Render a file offline through the real-time path, in the engine's hearing configuration.

    Channel 1 of the file is input 1 and channel 2 is input 2, routed as set on the engine; a mono file feeds
    both inputs. The result is a stereo file [left, right] at the file's own sample rate. With trim=True the
    processing delay is removed, so the output has the input's length and is aligned with it to about 1 ms.
    With trim=False the delay is kept, as heard live."""
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(src, always_2d=True)
    if x.shape[1] > 2:
        print(f"warning: {src} has {x.shape[1]} channels; using the first two")
        x = x[:, :2]
    n_in = x.shape[0]
    if sr != eng.fs:
        g = gcd(int(sr), eng.fs)
        x = resample_poly(x, eng.fs // g, int(sr) // g, axis=0)
    if x.shape[1] == 1 and eng.routing != "one":
        x = np.repeat(x, 2, axis=1)                         # mono file: the same signal at both microphones
    pad = int(0.2 * eng.fs) + eng.blocksize                 # room for the processing delay and the last block
    x = np.concatenate((x, np.zeros((pad, x.shape[1]))))
    t = time.perf_counter()
    y = E.process_file_multi(eng, x)
    dt = time.perf_counter() - t
    if trim:
        side = eng.ci_side if eng.config != "bilateral" else "left"
        d = int(round(eng.chains[side].latency_ms() * 1e-3 * eng.fs))
        y = y[d:]
    if sr != eng.fs:
        g = gcd(int(sr), eng.fs)
        y = resample_poly(y, int(sr) // g, eng.fs // g, axis=0)
    y = y[:n_in] if trim else y
    sf.write(dst, y.astype(np.float32), int(sr))
    dur = n_in / sr
    print(f"{src} -> {dst}: {dur:.1f} s, {eng.config}{'' if eng.config == 'bilateral' else ' ' + eng.ci_side}, "
          f"{eng.proc.kind} vocoder, in {dt:.1f} s (x{dur / dt:.1f} real time)")


def voc_opts(kind, a):
    if kind == "gpyt":
        return {}
    o = dict(env_lp=a.env_lp)
    if kind == "get":
        o["pulse_rate"] = a.get_rate
    return o


# ------------------------------------------------------------------------------------------------ terminal mode
class _Keys:
    """Non-blocking single-key reader: msvcrt on Windows, cbreak-mode stdin elsewhere."""

    def __enter__(self):
        if sys.platform.startswith("win"):
            import msvcrt
            self._m = msvcrt
        else:
            import termios
            import tty
            self._m = None
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._m is None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def keys(self):
        out = []
        if self._m is not None:
            while self._m.kbhit():
                out.append(self._m.getwch())
        else:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                out.append(sys.stdin.read(1))
        return out


def run_terminal(eng, a):
    eng.gate_enabled, eng.gate_margin_db = a.gate, a.gate_margin
    eng.config, eng.ci_side = a.config, a.ci_side
    eng.routing, eng.mic_channel = (a.routing, 0) if a.routing in ("1L2R", "1R2L") else ("one", 0 if a.routing == "one1" else 1)
    eng.start(find_device(a.indev, "input", a.hostapi), find_device(a.outdev, "output", a.hostapi),
              exclusive=a.exclusive)
    print(f"running: in = {eng.devices[0]} | out = {eng.devices[1]}")
    print("keys: [c] CI  [n] natural  [m] mix  [v] vocoder  [b] bilateral  [u] unilateral  [d] SSD  [k] swap CI ear  "
          "[g] quiet gate  [a] AGC  [r] NR  [s] steering  [f] fine-structure  [+/-] volume  [q] quit")
    try:
        with _Keys() as kb:
            _terminal_loop(eng, a, kb)
    finally:
        eng.stop()
        print()


def _terminal_loop(eng, a, kb):
    while True:
        for k in kb.keys():
            k = k.lower()
            st = eng.strategy
            if k == "q":
                return
            elif k in "cnm":
                eng.mode = {"c": "ci", "n": "natural", "m": "mix"}[k]
            elif k in "bud":
                eng.config = {"b": "bilateral", "u": "unilateral", "d": "ssd"}[k]
            elif k == "k":
                eng.ci_side = "right" if eng.ci_side == "left" else "left"
            elif k == "g":
                eng.gate_enabled = not eng.gate_enabled
            elif k == "a":
                for s_ in eng.strategies:
                    s_.agc_enabled = not st.agc_enabled
            elif k == "r":
                for s_ in eng.strategies:
                    s_.nr_enabled = not st.nr_enabled
            elif k == "s":
                for s_ in eng.strategies:
                    s_.steering = "fixed" if st.steering == "f120" else "f120"
            elif k == "f":
                for s_ in eng.strategies:
                    s_.carrier_mode = 0 if st.carrier_mode else 1
            elif k == "v":
                ks = ["gpyt", "noise", "sine", "get"]
                nk = ks[(ks.index(eng.proc.kind) + 1) % len(ks)]
                eng.set_vocoder(nk, **voc_opts(nk, a))
            elif k in "+=":
                eng.volume_db += 2
            elif k == "-":
                eng.volume_db -= 2
        st = eng.strategy
        dev, algo, fifo = eng.latency_ms()
        cfg = eng.config if eng.config == "bilateral" else f"{eng.config}-{eng.ci_side[0].upper()}"
        print(f"\r{cfg:<12} {eng.proc.kind:<5} {eng.mode:<7} in {eng.level_spl:5.1f} dB SPL  vol {eng.volume_db:+.0f} dB  NR {'on ' if st.nr_enabled else 'off'}"
              f" steer {st.steering:<5} fs {'on ' if st.carrier_mode else 'off'}  cpu {100 * eng.cpu:4.1f}%"
              f"  latency ~{dev + algo + fifo:4.0f} ms  underruns {eng.underruns} xruns {eng.xruns}   ", end="")
        if eng.errors:
            print(f"\n  ERROR: {eng.errors.popleft()}")
        if eng.stream is not None and not eng.stream.active:
            print("\naudio stream stopped (device unplugged or driver error)")
            return
        time.sleep(0.2)


# ------------------------------------------------------------------------------------------------ GUI
def _cmap(v):
    """Map values in 0..1 to RGB uint8 colors, from dark blue through purple and orange to yellow."""
    xs = [0, .25, .5, .75, 1]
    cs = np.array([[8, 8, 24], [60, 20, 110], [170, 40, 100], [240, 120, 30], [252, 240, 150]], float)
    return np.stack([np.interp(v, xs, cs[:, i]) for i in range(3)], -1).astype(np.uint8)


class App:
    W, ROW = 720, 11                       # electrodogram view: 720 columns (~7 s), 16 rows

    def __init__(self, eng, a):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.eng, self.a = eng, a
        self._ready = False
        self.root = root = tk.Tk()
        root.title("Real-time CI simulator - AB HiRes Fidelity 120 (GPyT)")
        root.protocol("WM_DELETE_WINDOW", self.quit)
        pad = dict(padx=6, pady=3)

        # devices
        fd = ttk.LabelFrame(root, text="Audio devices (a Steinberg/Yamaha interface is selected automatically; avoid Bluetooth hands-free)")
        fd.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)
        self.api = tk.StringVar(value=a.hostapi)
        ttk.Label(fd, text="Host API").grid(row=0, column=0, sticky="w")
        apis = sorted({d["api"] for d in list_devices()})
        cb = ttk.Combobox(fd, textvariable=self.api, values=apis, width=18, state="readonly")
        cb.grid(row=0, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda e: self.fill_devices())
        ttk.Label(fd, text="Mic").grid(row=1, column=0, sticky="w")
        ttk.Label(fd, text="Speakers").grid(row=2, column=0, sticky="w")
        self.inv, self.outv = tk.StringVar(), tk.StringVar()
        self.in_cb = ttk.Combobox(fd, textvariable=self.inv, width=62, state="readonly")
        self.out_cb = ttk.Combobox(fd, textvariable=self.outv, width=62, state="readonly")
        self.in_cb.grid(row=1, column=1, columnspan=3, sticky="w")
        self.out_cb.grid(row=2, column=1, columnspan=3, sticky="w")
        ttk.Label(fd, text="Block").grid(row=0, column=2, sticky="e")
        self.blk = tk.StringVar(value=str(a.block))
        ttk.Combobox(fd, textvariable=self.blk, values=["64", "128", "256", "512"], width=6, state="readonly").grid(row=0, column=3, sticky="w")
        self.excl = tk.BooleanVar(value=a.exclusive)
        ttk.Checkbutton(fd, text="WASAPI exclusive (low latency)", variable=self.excl).grid(row=0, column=5, sticky="w")
        ttk.Label(fd, text="Mics").grid(row=1, column=5, sticky="w")
        self.route = tk.StringVar(value=a.routing)
        fch = ttk.Frame(fd)
        fch.grid(row=2, column=5, sticky="w")
        for t, v in (("In1=L In2=R", "1L2R"), ("In1=R In2=L", "1R2L"), ("one mic: In1", "one1"), ("In2", "one2")):
            ttk.Radiobutton(fch, text=t, variable=self.route, value=v, command=self.apply).pack(side="left")
        self.run_btn = ttk.Button(fd, text="Start", command=self.toggle_run)
        self.run_btn.grid(row=0, column=4, rowspan=3, padx=8, ipady=10)
        self.fill_devices()
        # per-input meters show which input has a mic and whether its gain and phantom power are right
        self.meters = []
        fmt_ = ttk.Frame(fd)
        fmt_.grid(row=3, column=0, columnspan=6, sticky="w", pady=(4, 0))
        for k in range(2):
            ttk.Label(fmt_, text=f"Input {k + 1}").pack(side="left", padx=(0, 4))
            pb = ttk.Progressbar(fmt_, length=200, maximum=100)
            pb.pack(side="left")
            lab = ttk.Label(fmt_, width=11)
            lab.pack(side="left", padx=(4, 14))
            self.meters.append((pb, lab))
        self.mic_warn = tk.StringVar()
        ttk.Label(fd, textvariable=self.mic_warn, foreground="#b03000", width=110).grid(row=4, column=0, columnspan=6, sticky="w")

        # listening mode + levels
        fm = ttk.LabelFrame(root, text="Listen (space bar switches CI / natural)")
        fm.grid(row=1, column=0, sticky="nsew", **pad)
        self.mode = tk.StringVar(value=eng.mode)
        for i, (t, v) in enumerate((("CI vocoder", "ci"), ("Natural (mic)", "natural"), ("Mix", "mix"))):
            ttk.Radiobutton(fm, text=t, variable=self.mode, value=v, command=self.apply).grid(row=0, column=i, sticky="w")
        self.mix = tk.DoubleVar(value=eng.mix * 100)
        self._slider(fm, "CI share in mix %", self.mix, 0, 100, 1)
        self.gin = tk.DoubleVar(value=eng.in_gain_db)
        self._slider(fm, "Mic gain dB", self.gin, *E.IN_GAIN_RANGE_DB, 2)
        self.vol = tk.DoubleVar(value=eng.volume_db)
        self._slider(fm, "Volume dB", self.vol, -40, 12, 3)
        self.cal_btn = ttk.Button(fm, text="Calibrate to 65 dB SPL: talk normally for 5 s", command=self.calibrate)
        self.cal_btn.grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)

        # strategy + hooks
        fs_ = ttk.LabelFrame(root, text="Processing (live)")
        fs_.grid(row=1, column=1, sticky="nsew", **pad)
        st = eng.strategy
        self.agc = tk.BooleanVar(value=st.agc_enabled)
        self.nr = tk.BooleanVar(value=st.nr_enabled)
        self.steer = tk.BooleanVar(value=st.steering == "f120")
        self.fine = tk.BooleanVar(value=st.carrier_mode == 1)
        ttk.Checkbutton(fs_, text="AGC (dual-loop; when off, a fixed 38.8 dB gain without compression)", variable=self.agc, command=self.apply).grid(row=10, column=0, sticky="w")
        self.gate_on = tk.BooleanVar(value=a.gate)
        ttk.Checkbutton(fs_, text="Quiet-room gate (mutes the mic noise floor; key G)", variable=self.gate_on,
                        command=self.apply).grid(row=11, column=0, sticky="w")
        fg = ttk.Frame(fs_)
        fg.grid(row=12, column=0, sticky="w")
        ttk.Label(fg, text="   opens at noise floor +").pack(side="left")
        self.gate_margin = tk.DoubleVar(value=a.gate_margin)
        self.gm_lab = ttk.Label(fg, width=6)
        ttk.Scale(fg, from_=8, to=30, variable=self.gate_margin, length=140,
                  command=lambda _=None: (self.gm_lab.config(text=f"{self.gate_margin.get():.0f} dB"), self.apply())).pack(side="left", padx=4)
        self.gm_lab.config(text=f"{a.gate_margin:.0f} dB")
        self.gm_lab.pack(side="left")
        self.gate_info = tk.StringVar()
        ttk.Label(fs_, textvariable=self.gate_info, foreground="#305080", width=58).grid(row=13, column=0, sticky="w")
        root.bind("<g>", lambda e: (self.gate_on.set(not self.gate_on.get()), self.apply()))
        root.bind("<G>", lambda e: (self.gate_on.set(not self.gate_on.get()), self.apply()))
        ttk.Checkbutton(fs_, text="AB noise reduction (stock GPyT NR)", variable=self.nr, command=self.apply).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(fs_, text="Current steering, 9 steps (F120 virtual channels)", variable=self.steer, command=self.apply).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(fs_, text="Fine-structure carrier (F120 peak-frequency modulation)", variable=self.fine, command=self.apply).grid(row=2, column=0, sticky="w")
        ttk.Label(fs_, text="Added blocks (hooks.py):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.hook_vars = []
        for h in st.audio_hooks + st.env_hooks:
            v = tk.BooleanVar(value=h.enabled)
            ttk.Checkbutton(fs_, text=h.name, variable=v, command=self.apply).grid(row=4 + len(self.hook_vars), column=0, sticky="w")
            self.hook_vars.append((h, v))

        # hearing configuration
        fc = ttk.LabelFrame(root, text="Hearing configuration (keys: B bilateral, U unilateral, D single-sided deafness, "
                                       "K swap CI side)")
        fc.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)
        self.cfg = tk.StringVar(value=a.config)
        for i, (t, v) in enumerate((("Bilateral CI", "bilateral"), ("Unilateral CI (other ear silent)", "unilateral"),
                                    ("SSD: CI + normal-hearing ear", "ssd"))):
            ttk.Radiobutton(fc, text=t, variable=self.cfg, value=v, command=self.apply).grid(row=0, column=i, sticky="w", padx=(0, 12))
        ttk.Label(fc, text="CI ear").grid(row=1, column=0, sticky="e")
        self.side = tk.StringVar(value=a.ci_side)
        fsd = ttk.Frame(fc)
        fsd.grid(row=1, column=1, sticky="w")
        for t, v in (("Left", "left"), ("Right", "right")):
            ttk.Radiobutton(fsd, text=t, variable=self.side, value=v, command=self.apply).pack(side="left")
        self.nh_gain = tk.DoubleVar(value=0.0)
        fnh = ttk.Frame(fc)
        fnh.grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(fnh, text="SSD normal-ear level relative to the CI ear (dB)").pack(side="left")
        self.nh_lab = ttk.Label(fnh, width=5)
        ttk.Scale(fnh, from_=-20, to=10, variable=self.nh_gain, length=180,
                  command=lambda _=None: (self.nh_lab.config(text=f"{self.nh_gain.get():+.0f}"), self.apply())).pack(side="left", padx=4)
        self.nh_lab.pack(side="left")
        self.nh_align = tk.BooleanVar(value=False)
        ttk.Checkbutton(fnh, text="Delay normal ear to match the CI processing delay", variable=self.nh_align,
                        command=self.apply).pack(side="left", padx=(16, 0))
        self.cfg_info = tk.StringVar()
        ttk.Label(fc, textvariable=self.cfg_info, foreground="#305080", width=110).grid(row=3, column=0, columnspan=3, sticky="w")
        for key, v in (("b", "bilateral"), ("u", "unilateral"), ("d", "ssd")):
            root.bind(f"<{key}>", lambda e, v=v: (self.cfg.set(v), self.apply()))
            root.bind(f"<{key.upper()}>", lambda e, v=v: (self.cfg.set(v), self.apply()))
        root.bind("<k>", lambda e: (self.side.set("right" if self.side.get() == "left" else "left"), self.apply()))
        root.bind("<K>", lambda e: (self.side.set("right" if self.side.get() == "left" else "left"), self.apply()))

        # all vocoders are driven by the same F120 stimulation; only the acoustic carrier differs
        from .vocoders_rt import KINDS
        fv = ttk.LabelFrame(root, text="Vocoder (key V cycles; all use the same F120 stimulation)")
        fv.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)
        self.voc = tk.StringVar(value=a.vocoder)
        for i, (k, t) in enumerate(KINDS.items()):
            ttk.Radiobutton(fv, text=t, variable=self.voc, value=k, command=self.apply_vocoder).grid(row=0, column=i, sticky="w", padx=(0, 10))
        ttk.Label(fv, text="Envelope low-pass (noise/sine/GET)").grid(row=1, column=0, sticky="e")
        self.env_lp = tk.StringVar(value=str(int(a.env_lp)))
        c1 = ttk.Combobox(fv, textvariable=self.env_lp, values=["50", "160", "300", "500"], width=6, state="readonly")
        c1.grid(row=1, column=1, sticky="w")
        c1.bind("<<ComboboxSelected>>", lambda e: self.apply_vocoder(force=True))
        ttk.Label(fv, text="GET pulse rate (pps)").grid(row=1, column=2, sticky="e")
        self.get_rate = tk.StringVar(value=str(int(a.get_rate)))
        c2 = ttk.Combobox(fv, textvariable=self.get_rate, values=["100", "200", "300", "500"], width=6, state="readonly")
        c2.grid(row=1, column=3, sticky="w")
        c2.bind("<<ComboboxSelected>>", lambda e: self.apply_vocoder(force=True))
        self._voc_applied = None
        root.bind("<v>", self.cycle_vocoder)
        root.bind("<V>", self.cycle_vocoder)

        # electrodogram
        fe = ttk.LabelFrame(root, text="Electrodogram")
        self.fe_label, self._fe_side = fe, None
        fe.grid(row=4, column=0, columnspan=2, sticky="ew", **pad)
        self.photo = tk.PhotoImage(width=self.W, height=16 * self.ROW)
        self._sweep_x, self._slow = 0, 0
        tk.Label(fe, image=self.photo, bd=0).pack()
        self.status = tk.StringVar(value="stopped")
        ttk.Label(root, textvariable=self.status, font=("Consolas", 9), width=150).grid(row=5, column=0, columnspan=2, sticky="w", **pad)
        self.perf_warn = tk.StringVar()
        ttk.Label(root, textvariable=self.perf_warn, foreground="#c00000", width=150).grid(row=6, column=0, columnspan=2, sticky="w", padx=6)
        root.bind("<space>", self.toggle_ab)
        self._ready = True
        self.apply()
        self.apply_vocoder()
        self._put(0, np.full((16 * self.ROW, self.W, 3), 8, np.uint8))   # draw the dark background once
        root.update_idletasks()
        root.geometry(f"{root.winfo_reqwidth()}x{root.winfo_reqheight()}")  # fix the window size so it does not jump
        self.root.after(50, self.tick)

    def _slider(self, parent, text, var, lo, hi, row):
        ttk = self.ttk
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w")
        lab = ttk.Label(parent, width=6)
        lab.grid(row=row, column=2, sticky="w")

        def cb(_=None):
            lab.config(text=f"{var.get():.0f}")
            self.apply()
        ttk.Scale(parent, from_=lo, to=hi, variable=var, command=cb, length=220).grid(row=row, column=1, sticky="w")
        cb()

    def fill_devices(self):
        devs = list_devices(self.api.get())
        self.devs_in = [d for d in devs if d["nin"] > 0]
        self.devs_out = [d for d in devs if d["nout"] > 0]
        fmt = lambda d, k: f"{d['index']}: {d['name']}  ({d[k]} ch, {d['fs']:.0f} Hz)"
        self.in_cb["values"] = [fmt(d, "nin") for d in self.devs_in]
        self.out_cb["values"] = [fmt(d, "nout") for d in self.devs_out]
        try:
            di = find_device(self.a.indev, "input", self.api.get())
            do = find_device(self.a.outdev, "output", self.api.get())
        except SystemExit:
            di = do = None
        for cbx, lst, want in ((self.in_cb, self.devs_in, di), (self.out_cb, self.devs_out, do)):
            idx = next((i for i, d in enumerate(lst) if d["index"] == want), 0)
            if lst:
                cbx.current(idx)

    def apply(self):
        if not self._ready:
            return
        e = self.eng
        e.mode = self.mode.get()
        r = self.route.get()
        e.routing, e.mic_channel = (r, 0) if r in ("1L2R", "1R2L") else ("one", 0 if r == "one1" else 1)
        e.config, e.ci_side = self.cfg.get(), self.side.get()
        e.nh_gain_db = self.nh_gain.get()
        e.nh_delay_ms = e.ci_delay_ms() if self.nh_align.get() else 0.0
        e.gate_enabled = self.gate_on.get()
        e.gate_margin_db = self.gate_margin.get()
        e.mix = self.mix.get() / 100
        e.in_gain_db = self.gin.get()
        e.volume_db = self.vol.get()
        for st in e.strategies:                                   # every ear gets the same strategy settings
            st.agc_enabled = self.agc.get()
            st.nr_enabled = self.nr.get()
            st.steering = "f120" if self.steer.get() else "fixed"
            st.carrier_mode = 1 if self.fine.get() else 0
        for i, (h, v) in enumerate(self.hook_vars):
            e.set_hook(i, v.get())
        self.cfg_info.set(self.describe())

    def describe(self):
        e = self.eng
        mic = {"left": "left mic", "right": "right mic"}
        if e.routing == "one":
            mic = {s: f"input {e.mic_channel + 1}" for s in ("left", "right")}
        else:
            iL = 1 if e.routing == "1L2R" else 2
            mic = {"left": f"input {iL}", "right": f"input {3 - iL}"}
        if e.config == "bilateral":
            if e.routing == "one":
                return f"Left and right ear: one CI processor on {mic['left']} (same signal in both ears)"
            return (f"Left ear: CI on {mic['left']}   |   Right ear: CI on {mic['right']}   "
                    "(two independent processors)")
        ci = e.ci_side
        oth = "right" if ci == "left" else "left"
        txt = f"{ci.capitalize()} ear: CI on {mic[ci]}   |   {oth.capitalize()} ear: "
        if e.config == "unilateral":
            return txt + "silent"
        d = f", delayed {e.nh_delay_ms:.0f} ms" if e.nh_delay_ms > 0 else ", no added delay (acoustic ear)"
        return txt + f"normal hearing, {mic[oth]} unvocoded, {e.nh_gain_db:+.0f} dB{d}"

    def apply_vocoder(self, force=False):
        kind = self.voc.get()
        opts = {} if kind == "gpyt" else dict(env_lp=float(self.env_lp.get()))
        if kind == "get":
            opts["pulse_rate"] = float(self.get_rate.get())
        if force or (kind, tuple(sorted(opts.items()))) != self._voc_applied:
            self.eng.set_vocoder(kind, **opts)
            self._voc_applied = (kind, tuple(sorted(opts.items())))

    def cycle_vocoder(self, _=None):
        from .vocoders_rt import KINDS
        ks = list(KINDS)
        self.voc.set(ks[(ks.index(self.voc.get()) + 1) % len(ks)])
        self.apply_vocoder()

    def toggle_ab(self, _=None):
        self.mode.set("natural" if self.mode.get() == "ci" else "ci")
        self.apply()

    def toggle_run(self):
        from tkinter import messagebox
        if self.eng.stream is not None:
            self.eng.stop()
            self.run_btn.config(text="Start")
            return
        if not self.devs_in or not self.devs_out:
            messagebox.showerror("No audio devices", f"No input and output devices found for host API "
                                 f"'{self.api.get()}'. Pick another host API in the list.")
            return
        try:
            di = self.devs_in[self.in_cb.current()]["index"]
            do = self.devs_out[self.out_cb.current()]["index"]
            bs = int(self.blk.get())
            if bs != self.eng.blocksize:
                self.eng.set_rate(self.eng.fs, bs)
            self.eng.start(di, do, exclusive=self.excl.get())
            self.apply()
            self._voc_applied = None
            self.apply_vocoder()
            self.run_btn.config(text="Stop")
        except Exception as ex:
            self.eng.stop()
            messagebox.showerror("Could not open audio devices", f"{ex}\n\nIf 'WASAPI exclusive' is checked, close other "
                                 "apps that use the interface, or uncheck it. Mic and speakers must use the same host API.")

    def calibrate(self):
        if self.eng.stream is None:
            self.status.set("start the audio first, then calibrate")
            return
        self.cal_btn.config(state="disabled", text="listening... talk normally (5 s)")

        def work():
            try:
                lvl, g = self.eng.calibrate(5.0)
            except Exception as ex:
                self.root.after(0, lambda: self.cal_btn.config(state="normal", text=f"Calibration failed: {ex}"))
                return
            self.root.after(0, lambda: (self.gin.set(g), self.apply(),
                                        self.cal_btn.config(state="normal", text=f"Calibrated: mic gain {g:+.1f} dB "
                                                            f"(speech was {lvl + E.SPL_OFFSET:.0f} dB SPL at 0 dB gain)")))
        threading.Thread(target=work, daemon=True).start()

    def _set(self, var, text):
        """Set a Tk variable only when its text changes, because every set forces a redraw and re-layout."""
        if var.get() != text:
            var.set(text)

    def tick(self):
        """Refresh the GUI.

        This must stay cheap because it shares Python's interpreter lock with the audio callback. A 23 ms
        full-image redraw stalls the 2.9 ms audio blocks, which causes dropouts (crackle) and makes the FIFO grow
        to absorb them (extra delay). The electrodogram is therefore drawn as a sweep every 50 ms, with only the
        new columns drawn at a moving cursor. Text and meters refresh four times per second, and only when they
        change."""
        e = self.eng
        cols = []
        while e.elec_cols:
            cols.append(e.elec_cols.popleft())
        if cols:
            c = np.array(cols).T[:, -(self.W - 2):]                # (16, n) newest columns
            n = c.shape[1]
            v = np.clip(c / 600.0, 0, 1) ** 0.6                    # clinical units: T=50, M=500
            rgb = np.repeat(_cmap(v[::-1]), self.ROW, axis=0)      # electrode 1 at the bottom
            rgb[self.ROW - 1::self.ROW] //= 3                      # row separators
            x = self._sweep_x
            first = min(n, self.W - x)
            parts = [(x, rgb[:, :first])]
            if first < n:
                parts.append((0, rgb[:, first:]))
            for x0, blk in parts:
                self._put(x0, blk)
            self._sweep_x = (x + n) % self.W
            cur = np.full((16 * self.ROW, 2, 3), 255, np.uint8)    # white cursor just ahead of the newest data
            self._put(min(self._sweep_x, self.W - 2), cur)
        self._slow += 1
        if self._slow % 5 == 0:
            self._slow_tick()
        self.root.after(50, self.tick)

    def _put(self, x, rgb):
        h, w = rgb.shape[:2]
        ppm = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb).tobytes()
        self.photo.tk.call(self.photo.name, "put", ppm, "-format", "ppm", "-to", int(x), 0)

    def _slow_tick(self):
        e = self.eng
        if e.stream is not None and e.ch_dbfs is not None:
            for k, (pb, lab) in enumerate(self.meters):
                if k < len(e.ch_dbfs):
                    v = float(e.ch_dbfs[k])
                    pb["value"] = max(0.0, min(100.0, v + 100))
                    lab.config(text=f"{v:6.1f} dBFS")
            nch = len(e.ch_dbfs)
            if e.routing == "one":
                need = [min(e.mic_channel, nch - 1)]
            elif e.config == "bilateral" or e.config == "ssd":
                need = list(range(min(2, nch)))
            else:
                need = [(0 if e.ci_side == "left" else 1) if e.routing == "1L2R" else (1 if e.ci_side == "left" else 0)]
            dead = [i + 1 for i in need if i < nch and e.ch_dbfs[i] < -85]
            self._set(self.mic_warn, "" if not dead else
                      f"No mic signal on input {', '.join(map(str, dead))}: turn up the input gain knob on the "
                      "UR22, switch on +48V for condenser mics, and check the MIC/LINE (HI-Z) switch.")
            if self.nh_align.get():
                e.nh_delay_ms = e.ci_delay_ms()
            self._set(self.cfg_info, self.describe())
            if e.gate_enabled:
                fl, th, op = e.gate_status()
                self._set(self.gate_info, f"   noise floor {fl:4.0f} dB SPL, opens above {th:4.0f}:  "
                          f"{'open (sound)  ' if op else 'closed (quiet)'}" if fl is not None else "")
            else:
                self._set(self.gate_info, "   gate off")
        side = "left" if e.config == "bilateral" else e.ci_side
        if self._fe_side != side:
            self._fe_side = side
            self.fe_label.config(text=f"Electrodogram, {side} CI (electrode 1, apical and low frequency, at the "
                                      "bottom; ~7 s sweep)")
        if e.stream is not None and not e.stream.active:          # stopped by the driver (device unplugged, error)
            e.stop()
            self.run_btn.config(text="Start")
            self._set(self.status, "audio stream stopped by the driver (device unplugged?). Press Start to restart.")
            return
        if e.stream is not None:
            dev, algo, fifo = e.latency_ms()
            err = f"  ERROR: {e.errors[-1]}" if e.errors else ""
            w = getattr(e, "wall", 0.0)
            self._set(self.perf_warn, "" if w < 0.9 else
                      f"Processing can't keep up ({100 * w:.0f}% of real time), so the output crackles. Use block 256 or "
                      "512, the sine or GET vocoder, or unilateral/SSD instead of bilateral. If this happens at settings "
                      "that worked before, the CPU is throttling from heat or a weak charger: let the laptop cool down and "
                      "use its original charger.")
            self._set(self.status, f"{e.config.upper():<11}{e.proc.kind.upper():<5}| in {e.level_spl:5.1f} dB SPL | "
                      f"CPU {100 * e.cpu:4.1f}% (peak block {100 * e.cpu_peak:3.0f}%) | latency ~{dev + algo + fifo:3.0f} ms "
                      f"(device {dev:.0f} + CI {algo:.0f} + buffer {fifo:.0f}) | "
                      f"dropouts {e.underruns}  xruns {e.xruns}{err}")

    def quit(self):
        self.eng.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    ap.add_argument("--in", dest="indev", help="input device index or name substring")
    ap.add_argument("--out", dest="outdev", help="output device index or name substring")
    ap.add_argument("--hostapi", default=None, help="host API (default: ASIO if a Steinberg/Yamaha ASIO driver is "
                    "installed, else WASAPI)")
    ap.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=True,
                    help="WASAPI exclusive mode (UR22mkII: ~25 ms device latency vs ~45 ms shared)")
    ap.add_argument("--config", default="bilateral", choices=["bilateral", "unilateral", "ssd"],
                    help="bilateral: CI both ears (independent processors); unilateral: CI one ear, other silent; "
                    "ssd: CI one ear, other ear hears its own mic unvocoded")
    ap.add_argument("--ci-side", default="left", choices=["left", "right"], help="CI ear for unilateral / ssd")
    ap.add_argument("--routing", default="1L2R", choices=["1L2R", "1R2L", "one1", "one2"],
                    help="mic routing: input 1 = left & input 2 = right (default), swapped, or one mic for both")
    ap.add_argument("--block", type=int, default=128, help="callback block size in samples (64..4096; default 128)")
    ap.add_argument("--gain", type=float, default=0.0, help="mic gain in dB (calibration)")
    ap.add_argument("--volume", type=float, default=-6.0, help="output volume in dB")
    ap.add_argument("--vocoder", default="gpyt", choices=["gpyt", "noise", "sine", "get"],
                    help="gpyt: GPyT electrodogram vocoder; noise (noise-band), sine and get (GET pulsatile) are driven "
                    "by the same F120 stimulation currents")
    ap.add_argument("--env-lp", type=float, default=300.0, help="envelope low-pass cutoff in Hz for noise/sine/GET")
    ap.add_argument("--get-rate", type=float, default=100.0, help="GET pulse rate (pps)")
    ap.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True,
                    help="quiet-room gate: mutes the mic noise floor so a silent room is silent (default on)")
    ap.add_argument("--gate-margin", type=float, default=18.0, help="gate opens this many dB above the tracked noise floor")
    ap.add_argument("--nogui", action="store_true", help="terminal mode, without the GUI")
    ap.add_argument("--file", help="offline: input audio file (mono or stereo)")
    ap.add_argument("--outfile", help="offline: output wav (stereo [left, right])")
    ap.add_argument("--no-trim", action="store_true", help="offline: keep the processing delay in the output")
    a = ap.parse_args()
    if not 16 <= a.block <= 4096:
        ap.error("--block must be between 16 and 4096 samples")
    if not a.get_rate > 0:
        ap.error("--get-rate must be positive")
    if not 0 < a.env_lp <= 4000:
        ap.error("--env-lp must be in (0, 4000] Hz")
    if a.outfile and not a.file:
        ap.error("--outfile requires --file")

    if a.list:
        for d in list_devices():
            print(f"{d['index']:3d}  in {d['nin']}  out {d['nout']}  {d['fs']:6.0f} Hz  [{d['api']}]  {d['name']}")
        return
    _load_engine()
    eng = E.Engine(FS, a.block)
    eng.in_gain_db, eng.volume_db = a.gain, a.volume
    eng.process(np.zeros(4800))                                    # JIT warm-up before the stream opens
    eng.reset()
    eng.set_vocoder(a.vocoder, **voc_opts(a.vocoder, a))
    eng.gate_enabled, eng.gate_margin_db = a.gate, a.gate_margin
    if a.file:
        eng.config, eng.ci_side = a.config, a.ci_side
        eng.routing, eng.mic_channel = ((a.routing, 0) if a.routing in ("1L2R", "1R2L")
                                        else ("one", 0 if a.routing == "one1" else 1))
        render_file(a.file, a.outfile or str(Path(a.file).with_suffix("")) + "_ci.wav", eng, trim=not a.no_trim)
        return
    a.hostapi = a.hostapi or default_hostapi()
    if a.nogui:
        run_terminal(eng, a)
    else:
        App(eng, a).run()


if __name__ == "__main__":
    main()
