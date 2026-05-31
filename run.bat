@echo off
:: Cherche pythonw.exe dans plusieurs emplacements pour eviter d'ouvrir une fenetre CMD

set "PYW="
for %%P in (
    "C:\Users\akala\AppData\Local\Programs\Python\Python311\pythonw.exe"
    "C:\Users\akala\AppData\Local\Programs\Python\Python312\pythonw.exe"
    "C:\Users\akala\AppData\Local\Programs\Python\Python310\pythonw.exe"
    "C:\Python311\pythonw.exe"
    "C:\Python312\pythonw.exe"
) do if exist %%P set "PYW=%%P"

if not defined PYW (
    for /f "delims=" %%i in ('where pythonw 2^>nul') do set "PYW=%%i"
)

if not defined PYW (
    :: Fallback : python avec console (au pire des cas)
    start "" /B python "%~dp0manager.py"
) else (
    start "" /B "%PYW%" "%~dp0manager.py"
)
exit /b
