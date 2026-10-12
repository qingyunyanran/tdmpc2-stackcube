@echo off
set PATH=E:\My_programs\anaconda\envs\diffcl10;E:\My_programs\anaconda\envs\diffcl10\Library\bin;C:\WINDOWS\System32;C:\WINDOWS
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUNBUFFERED=1
cd /d E:\munichi\tdmpc2_stackcube\tdmpc2
E:\My_programs\anaconda\envs\diffcl10\python.exe -u train_stackcube.py --steps 300000 --eval-frequency 10000 --eval-episodes 20 --exp-name cpu300k --save-frequency 10000 > E:\munichi\tdmpc2_stackcube\logs_run\cpu300k.log 2>&1
