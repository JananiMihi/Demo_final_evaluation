@echo off
cd /d "%~dp0webapp_full"
cd /d "c:\Users\ASUS TUF X506H\Downloads\demo\Demo"
py app.py
pause@echo off
cd /d "%~dp0"

if exist "app.py" (
    py app.py
) else (
    echo app.py was not found in:
    echo %cd%
    pause
    exit /b 1
)

pause