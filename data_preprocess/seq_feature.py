#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: seq_feature.py
    Description:
    
Created by YongBai on 2020/3/16 1:11 PM.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import csv
import logging
import multiprocessing as mp
import pysam
import sys
sys.path.append('..')
from utils import get_config


def gen_pileup_batch(in_cram_lst_fname, chr='21'):
    if not os.path.exists(in_cram_lst_fname):
        sys.exit('Error: {} not found.'.format(in_cram_lst_fname))
    aneu_conf = get_config()

    seq_out_root_dir = os.path.join(aneu_conf['data_dir']['data_root_dir'], 'corrgc_bams')

    seq_feature_conf = aneu_conf['seq_feature']
    compute_node = seq_feature_conf['compute_node']
    run_dir = seq_feature_conf['run_dir']

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    cmd_str = """
    echo "samtools mpileup -q 0 -Q 0 -f {0} -r {1} -B -C 50 -x -s {2} -o {3}" > {4}
    {5} {4}
    """
    ref_fname = aneu_conf['data_dir']['hg38_ref_fa']
    if not os.path.exists(ref_fname):
        sys.exit('Error: {} not found.'.format(ref_fname))

    all_df = pd.read_csv(in_cram_lst_fname, sep='\t')
    n_samples = len(all_df)
    n_submit = 0
    for row_idx, row in all_df.iterrows():
        sample_id = row['Sample_id']
        cram_full_fname = row['Cram_full_fname']

        sample_dir = os.path.join(seq_out_root_dir, sample_id)
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        cram_base_name = os.path.basename(cram_full_fname).rsplit('.', 1)[0]

        pileup_out_fname = os.path.join(sample_dir, cram_base_name + '.chr' + chr + '.pileup')
        if os.path.exists(pileup_out_fname):
            logging.info('BAM for_{}_exist, removing...'.format(sample_id))
            os.remove(pileup_out_fname)

        run_sh_fname = os.path.join(run_dir, 'run_pileup_' + sample_id + '.sh')

        i_cmd_str = cmd_str.format(ref_fname,
                                   'chr'+chr,
                                   cram_full_fname,
                                   pileup_out_fname,
                                   run_sh_fname,
                                   compute_node).format(run_dir)
        os.system(i_cmd_str)
        # print(i_cmd_str)
        logging.info('sumbmit {}/{} job for {}'.format(row_idx + 1, n_samples, sample_id))
        n_submit += 1
        # if row_idx == 0:
        #     break

    print('# of submit {}'.format(n_submit))


def seq_slide(seq_len, win_size, stride_size):
    """
    :param seq_len:
    :param win_size:
    :param stride_size:
    :return:
    """
    if seq_len < win_size:
        raise Exception("length of sequence less than win size when slide window.")
    n_win = int((seq_len - win_size) // stride_size) + 1

    seq_start_indices = range(0, n_win * stride_size, stride_size)
    end_start = np.max(seq_start_indices) + win_size
    remain_len = seq_len - end_start
    return np.array(seq_start_indices, dtype=int), end_start, remain_len


def mp_init(l):
    global lock
    lock = l


def cal_seq_feature_wrap(args):
    return cal_seq_feature_worker(*args)


def cal_seq_feature_worker(start_pos, end_pos, in_pileup_fname, in_gc_correct_bam_fname, chr):
    # lock.acquire()
    logging.info('running: {}-{}'.format(start_pos, end_pos))
    correct_gc_bam = pysam.AlignmentFile(in_gc_correct_bam_fname, "rb")
    rc = correct_gc_bam.count(contig='chr' + chr, start=start_pos, stop=end_pos)
    logging.info('rc: {}'.format(rc))
    if rc == 0:
        correct_gc_bam.close()
        # lock.release()
        return np.array([start_pos, end_pos, rc, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        col_names = ['CHR', 'START_POS', 'REF_BASE', 'READS', 'READ_BASE', 'BASEQ', 'MAPQ']
        col_types = [str, int, str, int, str, str, str]
        col_dtypes = dict(zip(col_names, col_types))

        # solve error: pandas.errors.ParserError:
        # Error tokenizing data. C error: EOF inside string starting at line 15945
        pileup_df = pd.read_csv(in_pileup_fname, sep='\t', header=None, names=col_names,
                                dtype=col_dtypes, encoding='utf-8', quoting=csv.QUOTE_NONE)

        pileup_df_filter = pileup_df[(pileup_df['START_POS'] < end_pos+1) & (pileup_df['START_POS'] >= start_pos+1)]
        baseq_str = ''.join(pileup_df_filter['BASEQ'].values)
        mapq_str = ''.join(pileup_df_filter['MAPQ'].values)

        baseq_arr = np.array([ord(x)-33 for x in list(baseq_str)])
        baseq_min = np.min(baseq_arr)
        baseq_max = np.max(baseq_arr)
        baseq_mean = np.mean(baseq_arr)
        baseq_std = np.std(baseq_arr)
        baseq_median = np.median(baseq_arr)
        baseq_skew = stats.skew(baseq_arr)
        baseq_kurtosis = stats.kurtosis(baseq_arr)

        mapq_arr = np.array([ord(x)-33 for x in list(mapq_str)])
        mapq_min = np.min(mapq_arr)
        mapq_max = np.max(mapq_arr)
        mapq_mean = np.mean(mapq_arr)
        mapq_std = np.std(mapq_arr)
        mapq_median = np.median(mapq_arr)
        mapq_skew = stats.skew(mapq_arr)
        mapq_kurtosis = stats.kurtosis(mapq_arr)

        correct_gc_bam.close()
        # lock.release()
        return np.array([start_pos, end_pos, rc,
                         baseq_min, baseq_max, baseq_mean, baseq_std, baseq_median, baseq_skew, baseq_kurtosis,
                         mapq_min, mapq_max, mapq_mean, mapq_std, mapq_median, mapq_skew, mapq_kurtosis])


def cal_seq_feature(in_pileup_fname, in_gc_correct_bam_fname, out_fname,
                    chr='21', ref_chr_len=46709983, bin_size=1000000, step_size=500000, n_proc=10):

    # create bins

    start_pos_arr, end_start, end_len = seq_slide(ref_chr_len, bin_size, step_size)
    start_bin_arr = [(x, x + bin_size, in_pileup_fname, in_gc_correct_bam_fname, chr) for x in start_pos_arr]
    if end_start < ref_chr_len and end_len > 0:
        start_bin_arr.append((end_start, ref_chr_len, in_pileup_fname, in_gc_correct_bam_fname, chr))

    n_bin = len(start_bin_arr)
    re_array = np.zeros((n_bin, 17))
    logging.info('re_array shape: {}'.format(re_array.shape))

    # locker = mp.Lock()
    # cal_feature_p = mp.Pool(n_proc, initializer=mp_init, initargs=(locker,))
    cal_feature_p = mp.Pool(n_proc)
    cal_feature_results = cal_feature_p.imap(cal_seq_feature_wrap, start_bin_arr)
    for i, re in enumerate(cal_feature_results):
        logging.info('finished at {}-{}'.format(start_bin_arr[i][0], start_bin_arr[i][1]))
        re_array[i, :] = re
    out_cols = ['START_POS', 'END_POS', 'RC',
                'BASEQ_MIN', 'BASEQ_MAX', 'BASEQ_MEAN', 'BASEQ_STD', 'BASEQ_MEDIAN', 'BASEQ_SKEW', 'BASEQ_KURTOSIS',
                'MAPQ_MIN', 'MAPQ_MAX', 'MAPQ_MEAN', 'MAPQ_STD', 'MAPQ_MEDIAN', 'MAPQ_SKEW', 'MAPQ_KURTOSIS']
    # col_types = [int] * 3 + [float] * 14
    # col_dtype = dict(zip(out_cols, col_types))
    out_df = pd.DataFrame(data=re_array, columns=out_cols, dtype=float)
    out_df['START_POS'] = out_df['START_POS'].astype(int)
    out_df['END_POS'] = out_df['END_POS'].astype(int)
    out_df['RC'] = out_df['RC'].astype(int)

    out_df.to_csv(out_fname, sep='\t', index=False)

    cal_feature_p.close()
    cal_feature_p.join()
    logging.info('Done, result saved at {}'.format(out_fname))


def load_seq_feature(in_feature_fname):
    # feat_list, ignore_rows
    feat_list = ['RC', 'BASEQ_MEAN', 'BASEQ_STD', 'BASEQ_SKEW', 'BASEQ_KURTOSIS', 'MAPQ_MEAN', 'MAPQ_STD', 'MAPQ_SKEW', 'MAPQ_KURTOSIS']
    # feat_list = ['RC', 'BASEQ_MEAN', 'BASEQ_STD', 'MAPQ_MEAN', 'MAPQ_STD']
    skiprows = [i for i in range(1, 9)]  # skip the first 8 rows because there does nothave read count in chr 21

    return pd.read_csv(in_feature_fname, sep='\t', usecols=feat_list, skiprows=skiprows).values


