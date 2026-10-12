@echo off
set PATH=E:\My_programs\anaconda\envs\diffcl10;E:\My_programs\anaconda\envs\diffcl10\Library\bin;C:\WINDOWS\System32;C:\WINDOWS
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUNBUFFERED=1
cd /d E:\munichi\tdmpc2_stackcube\tdmpc2
E:\My_programs\anaconda\envs\diffcl10\python.exe -u train_stackcube.py --steps 100000 --num-samples 512 --horizon 5 --eval-frequency 10000 --eval-episodes 30 --exp-name seed1_warmonline3 --save-frequency 10000 --warm-start checkpoints\seed1_seed1_warmonline2\best.pt > E:\munichi\tdmpc2_stackcube\logs_run\warmonline3.log 2>&1
