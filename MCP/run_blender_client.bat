@echo off
REM Batch file to run Blender MCP Client with correct Python

echo ========================================
echo Blender MCP Client Launcher
echo ========================================
echo.

REM Try different Python commands
echo Checking for Python installation...

REM Method 1: Try py launcher (Windows Python Launcher)
where py >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python launcher 'py'
    py --version
    echo.
    echo Installing/updating anthropic...
    py -m pip install --upgrade anthropic
    echo.
    echo Starting Blender MCP Client...
    py blender_mcp_client.py
    goto :end
)

REM Method 2: Try python3
where python3 >nul 2>&1
if %errorlevel% == 0 (
    echo Found 'python3'
    python3 --version
    echo.
    echo Installing/updating anthropic...
    python3 -m pip install --upgrade anthropic
    echo.
    echo Starting Blender MCP Client...
    python3 blender_mcp_client.py
    goto :end
)

REM Method 3: Try python
where python >nul 2>&1
if %errorlevel% == 0 (
    echo Found 'python'
    python --version
    echo.
    echo Installing/updating anthropic...
    python -m pip install --upgrade anthropic
    echo.
    echo Starting Blender MCP Client...
    python blender_mcp_client.py
    goto :end
)

REM Method 4: Try direct path to Windows Store Python
set WINSTORE_PYTHON=C:\Users\%USERNAME%\AppData\Local\Microsoft\WindowsApps\python.exe
if exist "%WINSTORE_PYTHON%" (
    echo Found Windows Store Python
    "%WINSTORE_PYTHON%" --version
    echo.
    echo Installing/updating anthropic...
    "%WINSTORE_PYTHON%" -m pip install --upgrade anthropic
    echo.
    echo Starting Blender MCP Client...
    "%WINSTORE_PYTHON%" blender_mcp_client.py
    goto :end
)

echo ERROR: Python not found!
echo Please install Python from https://www.python.org or Microsoft Store
echo.
pause
goto :end

:end
echo.
pause