#!/usr/bin/env sh

# =====================================================
# Description: run.sh.sh
#
# =====================================================
#
# Created by YongBai on 2020/3/2 2:30 PM.

echo "python main.py">run_script.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=40g,p=1 run_script.sh
#qsub -cwd -P P18Z10200N0124 -q st.q -l vf=10g,p=1 run_script.sh