#!/bin/zsh
# AI 410 app-demo video: resets the local HOS 4 app so a recording starts from the five
# store documents, without the rental sheet the video adds on screen.
# Usage: ./reset-app.sh <app-dir>   (the API, worker, Redis, web chat and Expo web keep running)
set -euo pipefail
APP=${1:?app dir}
cd $APP/backend
rm -rf chroma_db tool-calls.log
./venv/bin/python ingest.py
