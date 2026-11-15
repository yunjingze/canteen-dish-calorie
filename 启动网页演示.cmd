@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo.
echo   Starting the canteen dish demo from:
echo   %CD%
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_demo.ps1" %*
if errorlevel 1 (
  echo.
  echo   The demo exited with an error. Check the message above.
  pause
)
