@echo off
setlocal
cd /d "%~dp0"
title StatsPro - make a Pro key

if not exist ".license_secret" (
  echo First time setup: paste your LICENSE_SECRET and press Enter.
  echo It is saved only on this computer, in the file .license_secret
  set /p SECRET=Secret: 
  >".license_secret" echo %SECRET%
  echo.
)
set /p LICENSE_SECRET=<".license_secret"

set DAYS=
set /p DAYS=How many days? Type 7 or 30 then press Enter [7]: 
if "%DAYS%"=="" set DAYS=7

echo.
for /f "delims=" %%K in ('python tools\make_key.py --days %DAYS%') do set KEY=%%K
if "%KEY%"=="" (
  echo Could not make a key. Check that Python is installed and the secret is correct.
  pause
  exit /b 1
)
echo Your key for %DAYS% days:
echo.
echo     %KEY%
echo.
<nul set /p=%KEY%| clip
echo The key was copied. Paste it into your email or WhatsApp reply to the buyer.
echo.
pause
