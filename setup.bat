@echo off
:: CF Rating Predictor — Windows Setup & Launcher
:: Usage: setup.bat [collect|train|serve|all]

setlocal
:: Try venv first, then fall back to system Python 3.14
set VENV_PYTHON=venv\bin\python.exe
if not exist %VENV_PYTHON% set VENV_PYTHON=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe
if not exist %VENV_PYTHON% set VENV_PYTHON=python

if "%1"=="" goto :serve
if "%1"=="collect" goto :collect
if "%1"=="train" goto :train
if "%1"=="serve" goto :serve
if "%1"=="all" goto :all
if "%1"=="install" goto :install

:install
echo Installing dependencies...
%VENV_PYTHON% -m pip install -r requirements.txt
echo.
echo Done! Run setup.bat collect to fetch data.
goto :end

:collect
echo Collecting Codeforces problems (this may take 30-60 minutes)...
%VENV_PYTHON% src/data_collection/scraper.py --output data/raw/problems.jsonl %~2 %~3 %~4
goto :end

:train
echo Training ML models...
%VENV_PYTHON% src/models/train.py --data data/raw/problems.jsonl --output models/ %~2 %~3
goto :end

:serve
echo Starting web server...
echo Open browser at: http://localhost:5000
%VENV_PYTHON% src/api/app.py --port 5000
goto :end

:all
echo === Full Pipeline: Collect → Train → Serve ===
call :collect
call :train
call :serve
goto :end

:end
endlocal
