@echo off
REM Windows Environment Initialization Script

echo Initializing project environment...

REM Set PYTHONPATH environment variable
set PYTHONPATH=%cd%\src
echo PYTHONPATH has been set to: %PYTHONPATH%

REM Activate virtual environment (if it exists)
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo.
echo Environment initialization completed!
echo You can now run project commands.
echo.
echo Examples:
echo   Run RAG module: python src\RAG\rag.py
echo   Start development server: langgraph dev
echo.

pause