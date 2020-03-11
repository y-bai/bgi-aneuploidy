#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: main.py
    Description:
    
Created by YongBai on 2020/3/2 2:22 PM.
"""
import os
import logging
from data_preprocess import get_seq_lst, get_sample_info, get_tn_info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':

#     test_cram = "/zfssz6/ST_MCHRI/BIGDATA/Million_seq500/subsample_database/CL100093941_L01_12/CL100093941_L01_12.sorted.rmdup.realign.BQSR.cram"
#     ref_2bit = "/zfssz6/ST_MCHRI/BIGDATA/database/BGI-seq500_OSS_download/human_reference/hg38/Homo_sapiens_assembly38.2bit"
#     out_freq = "/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_predict/freq_gc.txt"
#     out_plot = "/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_predict/gc.png"
#     out_corr = "/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_predict/out_chr21_sub.bw"
#
#     cmd_str = """
#     echo "correctGCBias -b {0} \
# --effectiveGenomeSize 2913022398 \
# -g {1} \
# -p 20 \
# -freq {2} \
# --binSize 50000 -o {3} -r chr21" > run_script.sh
# qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=40g,p=1 run_script.sh
#     """.format(test_cram, ref_2bit, out_freq, out_corr)
#     os.system(cmd_str)

#     get_seq_lst(reload=False, seq_type='fn', chr='18', in_fname='million.Positive.False.18.recheck.with.bampath.list')
# get_sample_info(reload=False, seq_type='tp', chr='21')

    get_tn_info()
    print('Done')