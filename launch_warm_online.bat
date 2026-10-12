@echo off
set PATH=E:\My_programs\anaconda\envs\diffcl10;E:\My_programs\anaconda\envs\diffcl10\Library\bin;C:\WINDOWS\System32;C:\WINDOWS
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUNBUFFERED=1
cd /d E:\munichi\tdmpc2_stackcube\tdmpc2
E:\My_programs\anaconda\envs\diffcl10\python.exe -u train_stackcube.py --warm-start checkpoints/warmstart.pt --steps 150000 --eval-frequency 10000 --eval-episodes 20 --exp-name warmonline --save-frequency 10000 > E:\munichi\tdmpc2_stackcube\logs_run\warmonline.log 2>&1
