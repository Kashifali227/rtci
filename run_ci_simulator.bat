@echo off
rem Windows launcher. Runs the Python that has rtci installed. Set RTCI_PYTHON to use another one, for example
rem     set RTCI_PYTHON=py -3.13
cd /d "%~dp0"
if defined RTCI_PYTHON (
    %RTCI_PYTHON% -m rtci %*
) else (
    python -m rtci %*
)
if errorlevel 1 pause
