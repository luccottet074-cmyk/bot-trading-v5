@echo off
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\lucco\OneDrive\Documents\entrainement python\V3_projet_bot_trading"
start "" "http://localhost:8503"
C:\Users\lucco\.conda\envs\finance\python.exe -m streamlit run dashboard/app.py --server.port 8503
