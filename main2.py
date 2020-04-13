#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: main2.py
    Description:
    
Created by YongBai on 2020/3/12 9:02 AM.
"""
import re
import numpy as np
import pandas as pd
import logging
from data_preprocess import *
from utils import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def extract_id(in_fname):
    """
    nohup.out.correct content:

    Your job 912685 ("run_cor_17B5087503.sh") has been submitted
    2020-03-12 02:08:32,242 - root - INFO - sumbmit 480/1900 job for 17B5087503
    Your job 912686 ("run_cor_18B2633219.sh") has been submitted
    2020-03-12 02:08:32,289 - root - INFO - sumbmit 481/1900 job for 18B2633219
    Unable to run job: job rejected: only 2000 jobs are allowed per user (current job count: 2000)
    Exiting.
    2020-03-12 02:08:32,349 - root - INFO - sumbmit 482/1900 job for 18B2670459
    Unable to run job: job rejected: only 2000 jobs are allowed per user (current job count: 2000)
    Exiting.

    :param in_fname: nohup.out.correct
    :return:
    """

    submiited_samples = []
    with open(in_fname, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            if line.startswith('Your job'):
                line = f.readline()
                search_g = re.search(r".*job for ([A-Z0-9].*)", line)
                if search_g:
                    submiited_samples.append(search_g.group(1))

    return submiited_samples


def extract_id2(in_fname):
    submitted_array = []
    with open(in_fname, 'r') as f:
        for line in f:
            search_g = re.search(r".*_([A-Z0-9].*)\.sh.*", line)
            if search_g:
                submitted_array.append(search_g.group(1))
    return np.unique(np.array(submitted_array))


def extract_id3(in_fname):
    out_df = pd.read_csv(in_fname, sep='\t', header=0, names=['f_size', 'f_name'])
    out_df['sample_id'] = out_df['f_name'].str.extract(pat='.*_([A-Z0-9].*)\.sh.*')
    out_df['f_size'] = out_df['f_size'].astype(int)
    out_df_t = out_df.copy()
    out_df_1 = out_df_t.groupby('sample_id').count().reset_index()
    out_df_1 = out_df_1[out_df_1['f_size'] == 1]
    out_df_1.sort_values(by=['sample_id'], inplace=True)
    out = pd.merge(out_df, out_df_1, how='inner', on='sample_id')
    out['o_type']=out['f_name_x'].str.extract(pat='.*sh\.([e|o])[0-9].*')
    out.to_csv('sh.submit.size.sampleid4', sep='\t', index=False)
    print(len(out))


if __name__ == '__main__':

    """
    nohup.out.correct content:
    
    Your job 912685 ("run_cor_17B5087503.sh") has been submitted
    2020-03-12 02:08:32,242 - root - INFO - sumbmit 480/1900 job for 17B5087503
    Your job 912686 ("run_cor_18B2633219.sh") has been submitted
    2020-03-12 02:08:32,289 - root - INFO - sumbmit 481/1900 job for 18B2633219
    Unable to run job: job rejected: only 2000 jobs are allowed per user (current job count: 2000)
    Exiting.
    2020-03-12 02:08:32,349 - root - INFO - sumbmit 482/1900 job for 18B2670459
    Unable to run job: job rejected: only 2000 jobs are allowed per user (current job count: 2000)
    Exiting.
    """
    # submit_job_nohup = 'nohup.out.correct'
    # submit_job_f = 'sh.submiited'
    # samples = extract_id(submit_job_nohup)
    # samples = extract_id2(submit_job_f)
    # for s in samples:
    #     print(s)
    # extract_id3('sh.submit.size')
    # file ='/zfssz6/ST_MCHRI/BIGDATA/Million_seq500/subsample_database/CL100093941_L01_12/CL100093941_L01_12.sorted.rmdup.realign.BQSR.cram'
    in_pileup_fname = '/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/corrgc_bams/18B3831885/CL100093941_L01_12.sorted.rmdup.realign.BQSR.chr21.pileup'
    in_gc_correct_bam_fname = '/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/corrgc_bams/18B3831885/CL100093941_L01_12.sorted.rmdup.realign.BQSR.gccorrect.chr21.bam'

    cal_seq_feature_worker(4500000, 5500000, in_pileup_fname, in_gc_correct_bam_fname, '21')
    # cal_seq_feature(file)
    # print(seq_slide(10, 5, 5))

