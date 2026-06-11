@echo off
REM One-click launcher for Windows - double-click in Explorer.
REM Starts the Streamlit chatbot after the one-time setup (venv + deps + .env).
cd /d "%~dp0"

echo ============================================
echo   FinDocRAG - Financial Document Intelligence (RAG)
echo ============================================

REM 1. Activate the project virtualenv created during setup.
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo.
  echo No .venv found. Run the one-time setup first:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

REM 2. Sanity-check secrets (warn, don't block).
if not exist ".env" (
  echo.
  echo WARNING: no .env found. Copy .env.example to .env and add your API keys
  echo          ^(OPENAI_API_KEY, ANTHROPIC_API_KEY, PINECONE_API_KEY^).
  echo.
)

REM 3. Make sure the corpus is built; if not, point the user at the build step.
if not exist "data\index\semantic\chunks.json" if not exist "data\index\fixed\chunks.json" (
  echo.
  echo No index found yet. Build the corpus once before launching:
  echo   python -m scripts.build_index
  echo.
  pause
  exit /b 1
)

REM 4. Launch. Streamlit opens the browser automatically; Ctrl+C stops it.
echo.
echo Launching the app... a browser tab will open. Press Ctrl+C here to stop.
echo.
streamlit run app.py

REM Keep the window open if Streamlit exits/errors.
echo.
pause
