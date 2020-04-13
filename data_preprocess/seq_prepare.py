#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: seq_prepare.py
    Description:
    
Created by YongBai on 2020/3/11 10:58 AM.
"""

import os
import numpy as np
import pandas as pd
import logging
from .data_prepare import _load_samples
import sys
sys.path.append('..')
from utils import get_config


def get_unique_cram_lst(chunk_size=1900):

    aneu_conf = get_config()
    data_root_dir = aneu_conf['data_dir']['data_root_dir']

    # get tp, fp, fn
    seq_types = ['tp', 'fp', 'fn']
    chrs = ['21', '18', '13']
    info_fnames = [seq_type + '.' + chr + '.million.all.info.list' for seq_type in seq_types for chr in chrs]
    info_fnames.append('tn.million.4train.info.list')

    all_df = None
    for info_f in info_fnames:
        info_full_f = os.path.join(data_root_dir, info_f)
        if not os.path.exists(info_full_f):
            continue
        logging.info('getting .cram file names from {}'.format(info_full_f))
        info_df = _load_samples(info_full_f)
        info_df['Cram_full_fname'] = info_df.apply(
            lambda x: os.path.join(x['Cram_file_path'], x['Cram']), axis=1)
        info_df = info_df[['Sample_id', 'Cram_full_fname']]

        if all_df is None:
            all_df = info_df
        else:
            all_df = pd.concat([all_df, info_df], axis=0)
    all_df.drop_duplicates(keep='first', inplace=True)
    all_df.reset_index(drop=True, inplace=True)
    n_samples = len(all_df)
    logging.info('finished loading all .cram file names for tp, fp, fn and tn..., unique N={}'.format(n_samples))
    # for job submit
    n_chunk = int(n_samples * 1.0 / chunk_size) + 1

    for i in range(n_chunk):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end >= n_samples:
            i_df = all_df.iloc[start:n_samples]
        else:
            i_df = all_df.iloc[start:end]
        out_fname = os.path.join(data_root_dir, 'all_trainsample_chunk_{}.cram.list'.format(i))
        if os.path.exists(out_fname):
            os.remove(out_fname)
        i_df.to_csv(out_fname, sep='\t', index=False)


def compute_gc_bias(in_cram_lst_fname):
    """

    :param in_cram_lst_fname:
    :return:
    """
    if not os.path.exists(in_cram_lst_fname):
        sys.exit('Error: {} not found.'.format(in_cram_lst_fname))
    aneu_conf = get_config()
    seq_out_root_dir = os.path.join(aneu_conf['data_dir']['data_root_dir'], 'corrgc_bams')

    deeptools_conf = aneu_conf['deeptools']
    ref_2bit = deeptools_conf['gc_hg38_ref_2bit']
    effective_genome_size = deeptools_conf['gc_effective_genome_size']
    n_processors = deeptools_conf['processors']
    fregment_length = deeptools_conf['fragment_len']
    run_dir = deeptools_conf['run_dir']
    compute_node = deeptools_conf['compute_node']

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    cmd_str = """
    echo "computeGCBias \
    -b {0} \
    --effectiveGenomeSize {1} \
    -g {2} \
    -l {3} \
    -p {4} \
    -o {5} " > {6}
    {7} {6}
    """

    all_df = pd.read_csv(in_cram_lst_fname, sep='\t')

    n_samples = len(all_df)

    for row_idx, row in all_df.iterrows():
        sample_id = row['Sample_id']
        cram_full_fname = row['Cram_full_fname']

        sample_dir = os.path.join(seq_out_root_dir, sample_id)
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        gc_freq_out_fname = os.path.join(sample_dir, 'gc_freq.txt')
        run_sh_fname = os.path.join(run_dir, 'run_' + sample_id + '.sh')
        i_cmd_str = cmd_str.format(cram_full_fname,
                                   effective_genome_size,
                                   ref_2bit,
                                   fregment_length,
                                   n_processors,
                                   gc_freq_out_fname,
                                   run_sh_fname,
                                   compute_node).format(run_dir)
        os.system(i_cmd_str)

        logging.info('sumbmit {}/{} job for {}'.format(row_idx + 1, n_samples, sample_id))


def correct_gc_bias(in_cram_lst_fname, chr='21'):
    """

    :param in_cram_lst_fname:
    :return:
    """

    if not os.path.exists(in_cram_lst_fname):
        sys.exit('Error: {} not found.'.format(in_cram_lst_fname))
    aneu_conf = get_config()
    seq_out_root_dir = os.path.join(aneu_conf['data_dir']['data_root_dir'], 'corrgc_bams')

    deeptools_conf = aneu_conf['deeptools']
    ref_2bit = deeptools_conf['gc_hg38_ref_2bit']
    effective_genome_size = deeptools_conf['gc_effective_genome_size']
    n_processors = deeptools_conf['processors']
    bin_size = deeptools_conf['bin_size']
    run_dir = deeptools_conf['run_dir']
    compute_node = deeptools_conf['compute_node']

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    cmd_str = """
    echo "correctGCBias \
    -b {0} \
    --effectiveGenomeSize {1} \
    -g {2} \
    -freq {3} \
    -p {4} \
    --binSize {5} \
    -r chr{6} \
    -o {7} " > {8}
    {9} {8}
    """

    all_df = pd.read_csv(in_cram_lst_fname, sep='\t')
    n_samples = len(all_df)
    n_submit = 0
    for row_idx, row in all_df.iterrows():
        sample_id = row['Sample_id']
        cram_full_fname = row['Cram_full_fname']

        sample_dir = os.path.join(seq_out_root_dir, sample_id)
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        gc_freq_out_fname = os.path.join(sample_dir, 'gc_freq.txt')
        if not os.path.exists(gc_freq_out_fname):
            sys.exit('Error: {} not found.'.format(gc_freq_out_fname))

        cram_base_name = os.path.basename(cram_full_fname).rsplit('.', 1)[0]

        corr_gc_out_fname = os.path.join(sample_dir, cram_base_name + '.gccorrect.chr' + chr + '.bam')
        if os.path.exists(corr_gc_out_fname):
            logging.info('BAM for_{}_exist.'.format(sample_id))
            continue
        run_sh_fname = os.path.join(run_dir, 'run_cor_' + sample_id + '.sh')
        i_cmd_str = cmd_str.format(cram_full_fname,
                                   effective_genome_size,
                                   ref_2bit,
                                   gc_freq_out_fname,
                                   n_processors,
                                   bin_size,
                                   chr,
                                   corr_gc_out_fname,
                                   run_sh_fname,
                                   compute_node).format(run_dir)
        os.system(i_cmd_str)

        logging.info('sumbmit {}/{} job for {}'.format(row_idx + 1, n_samples, sample_id))

        n_submit += 1
    print('# of submit {}'.format(n_submit))











