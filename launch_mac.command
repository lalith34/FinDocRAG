#!/usr/bin/env bash
# One-click launcher for macOS — double-click in Finder, or run ./launch_mac.command
# Starts the Streamlit chatbot after the one-time setup (venv + deps + .env).
cd "$(dirname "$0")" || exit 1

echo "============================================"
echo "  FinDocRAG — Financial Document Intelligence (RAG)"
echo "============================================"

# 1. Activate the project virtualenv created during setup.
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo
  echo "No .venv found. Run the one-time setup first:"
  echo "  python3.11 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  echo
  read -r -p "Press Enter to close..."
  exit 1
fi

# 2. Sanity-check secrets (warn, don't block).
if [ ! -f ".env" ]; then
  echo
  echo "WARNING: no .env found. Copy .env.example to .env and add your API keys"
  echo "         (ANTHROPIC_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, NEBIUS_API_KEY)."
  echo
fi

# 3. Make sure the corpus is built; if not, point the user at the build step.
if [ ! -f "data/index/semantic/chunks.json" ] && [ ! -f "data/index/fixed/chunks.json" ]; then
  echo
  echo "No index found yet. Build the corpus once before launching:"
  echo "  python -m scripts.build_index"
  echo
  read -r -p "Press Enter to close..."
  exit 1
fi

# 4. Launch. Streamlit opens the browser automatically; Ctrl+C stops it.
echo
echo "Launching the app… a browser tab will open. Press Ctrl+C here to stop."
echo
.venv/bin/streamlit run app.py

# Keep the Terminal window open if Streamlit exits/errors.
echo
read -r -p "App stopped. Press Enter to close..."
