@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\lucco\OneDrive\Documents\entrainement python\V3_projet_bot_trading"
C:\Users\lucco\.conda\envs\finance\python.exe -X utf8 -m execution.live_pipeline --precompute >> "C:\Users\lucco\OneDrive\Documents\entrainement python\V3_projet_bot_trading\logs\live_pipeline.log" 2>&1
