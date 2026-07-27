@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_summarize_rgb_f_pose.ps1"
if errorlevel 1 (
  echo.
  echo RGB-f pose summary failed.
  exit /b 1
)
endlocal
