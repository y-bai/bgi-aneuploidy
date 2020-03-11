#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: data_prepare.py
    Description:
    
Created by YongBai on 2020/3/10 4:31 PM.
"""
import os
import numpy as np
import pandas as pd
import sys
import logging
sys.path.append('..')
from utils import get_config


def get_seq_lst(reload=False, seq_type='tp', chr='21', in_fname=None):
    """
    processing files in folder: million.Insurance.database:
        million.Positive.True.21.recheck.with.bampath.list
        million.Positive.True.18.recheck.with.bampath.list
        million.Positive.True.13.recheck.with.bampath.list

        million.Positive.False.21.recheck.with.bampath.list
        million.Positive.False.18.recheck.with.bampath.list
        million.Positive.False.13.recheck.with.bampath.list
    :param reload
    :param seq_type:
    :param chr:
    :param in_fname:
    :return:
    """
    data_dir_conf = get_config()['data_dir']
    data_in_root = data_dir_conf['data_root_dir']

    out_pos_fname = os.path.join(data_in_root,  seq_type + '.' + chr + '.million.bampath.list')

    if reload:
        if not os.path.exists(out_pos_fname):
            sys.exit('Error: {} not found.'.format(out_pos_fname))
        else:
            return pd.read_csv(out_pos_fname, sep='\t')

    seq_data_path = os.path.join(data_in_root, data_dir_conf['input_pos_dir'])
    seq_tp_fname = os.path.join(seq_data_path, in_fname)
    mil_sample_fname = data_dir_conf['million_sample_lst']

    if not os.path.exists(seq_tp_fname):
        sys.exit('Error: {} not found.'.format(seq_tp_fname))
    if not os.path.exists(mil_sample_fname):
        sys.exit('Error: {} not found.'.format(mil_sample_fname))

    seq_tp_df = pd.read_csv(seq_tp_fname, sep='\t', header=None, usecols=[0], names=['Sample_id'])
    logging.info('# of samples: {}'.format(seq_tp_df.shape))
    if seq_tp_df.shape[0] > 0:
        milli_df = pd.read_csv(mil_sample_fname, sep='\t')
        seq_sample_id = seq_tp_df['Sample_id'].values

        out_df = milli_df[milli_df['Sample_id'].isin(seq_sample_id)]

        out_df.to_csv(out_pos_fname, sep='\t', index=False)
        return out_df
    else:
        sys.exit('Warning: {} does not have records'.format(seq_tp_fname))


def get_fp_seq_lst(chr='21', reload=False):

    """
    get NIPT false positive samples.
    input file: million.Negative.recheck.with.bampath.list
    outputs will to fp for chr 21, 18, 13
    :param chr:
    :param reload:
    :return:
    """
    conf = get_config()
    _input = conf['data_dir']
    data_root_dir = _input['data_root_dir']

    out_pos_fname = os.path.join(data_root_dir, 'fp.' + chr + '.million.bampath.list')
    if reload:
        re_df = pd.read_csv(out_pos_fname, sep='\t')
        logging.info('# of total samples of {}: {}'.format(chr, re_df.shape))
        return re_df
    else:
        milli_df = pd.read_csv(_input['million_sample_lst'], sep='\t')
        logging.info('# of million samples: {}'.format(milli_df.shape))
        # get 'chr'  list
        false_pos_files = os.path.join(data_root_dir, _input['input_fp_fname'])

        re_df = None
        in_df = pd.read_csv(false_pos_files, sep='\t', header=None,
                            usecols=[0, 1, 2], names=['Sample_id', 'Chr_annot', 'NIPT_label'])
        nipt_labels = in_df['NIPT_label'].str.split('/', expand=True)

        in_df['NIPT_T13'] = nipt_labels[0]
        in_df['NIPT_T18'] = nipt_labels[1]
        in_df['NIPT_T21'] = nipt_labels[2]
        in_df['truth'] = nipt_labels[3]

        if chr == '13':
            nipt_fp = in_df.loc[(in_df['NIPT_T13'] == 'T') & (in_df['truth'] == 'N'), 'Sample_id'].values
            milli_fp_df = milli_df[milli_df['Sample_id'].isin(nipt_fp)]
            logging.info('# of samples: {}'.format(milli_fp_df.shape))
            if os.path.exists(out_pos_fname):
                os.remove(out_pos_fname)
            milli_fp_df.to_csv(out_pos_fname, sep='\t', index=False)
            return milli_fp_df

        if chr == '18':
            nipt_fp = in_df.loc[(in_df['NIPT_T18'] == 'T') & (in_df['truth'] == 'N'), 'Sample_id'].values
            milli_fp_df = milli_df[milli_df['Sample_id'].isin(nipt_fp)]
            logging.info('# of samples: {}'.format(milli_fp_df.shape))
            if os.path.exists(out_pos_fname):
                os.remove(out_pos_fname)
            milli_fp_df.to_csv(out_pos_fname, sep='\t', index=False)
            return milli_fp_df

        if chr == '21':
            nipt_fp = in_df.loc[(in_df['NIPT_T21'] == 'T') & (in_df['truth'] == 'N'), 'Sample_id'].values
            milli_fp_df = milli_df[milli_df['Sample_id'].isin(nipt_fp)]
            logging.info('# of samples: {}'.format(milli_fp_df.shape))
            if os.path.exists(out_pos_fname):
                os.remove(out_pos_fname)
            milli_fp_df.to_csv(out_pos_fname, sep='\t', index=False)
            return milli_fp_df


def get_sample_info(reload=False, seq_type='tp', chr='21'):
    """
    :param reload
    :param seq_type: tp, fp, fn
    :param chr: 21, 18, 13
    :return:
    """

    data_dir_conf = get_config()['data_dir']
    data_in_root = data_dir_conf['data_root_dir']

    out_fname = os.path.join(data_in_root, seq_type + '.' + chr + '.million.all.info.list')
    if reload:
        if not os.path.exists(out_fname):
            sys.exit("Error: {} not found.".format(out_fname))

        sample_info_df = pd.read_csv(out_fname, sep='\t', dtype=str)
        sample_info_df['SAMPLE_AGE'] = sample_info_df['SAMPLE_AGE'].astype(float)
        sample_info_df['HEIGHT'] = sample_info_df['HEIGHT'].astype(float)
        sample_info_df['WEIGHT'] = sample_info_df['WEIGHT'].astype(float)
        sample_info_df['FF'] = sample_info_df['FF'].astype(float)

        return sample_info_df

    milli_info_fname = data_dir_conf['million_info_fname']
    seq_bam_fname = os.path.join(data_in_root, seq_type + '.' + chr + '.million.bampath.list')
    ff_fname = data_dir_conf['ff_info']

    if not os.path.exists(seq_bam_fname):
        sys.exit('Error: {} not found.'.format(seq_bam_fname))
    if not os.path.exists(milli_info_fname):
        sys.exit("Error: {} not found.".format(milli_info_fname))
    if not os.path.exists(ff_fname):
        sys.exit("Error: {} not found.".format(ff_fname))

    # load seq_lst
    seq_lst = pd.read_csv(seq_bam_fname, sep='\t')

    # load pat_info_df
    # Columns have mixed types. Specify dtype option on import or set low_memory=False.
    # So we simply read as str
    pat_info_df = pd.read_csv(milli_info_fname, sep='\t', dtype=str)

    # load fetal fraction
    ff_df_t = pd.read_csv(ff_fname, usecols=[0, 1])
    ff_df_t.columns = ['FF_Sample_id', 'FF']
    # same sample have multiple ff
    logging.info('unique samples in ff: {}'.format(ff_df_t['FF_Sample_id'].nunique()))
    ff_df = ff_df_t.groupby('FF_Sample_id')["FF"].agg('max').reset_index()

    seq_pat_info_df = pd.merge(seq_lst, pat_info_df, how='left', left_on=['Sample_id'], right_on=['SAMPLE_NUM'])
    logging.info('total number of {0} {1} samples in seq bam list= {2}, in patient info empty number is = {3}'.format(
        seq_type, chr, seq_lst.shape[0], seq_pat_info_df['SAMPLE_NUM'].isna().sum()))

    seq_pat_info_ff_df = pd.merge(seq_pat_info_df, ff_df, how='left', left_on=['Sample_id'], right_on=['FF_Sample_id'])

    logging.info('total number of {0} {1} samples in seq patient info = {2}, in ff list empty number is = {3}'.format(
        seq_type, chr, seq_pat_info_df.shape[0], seq_pat_info_ff_df['FF_Sample_id'].isna().sum()))

    seq_pat_info_ff_df['SAMPLE_AGE'] = seq_pat_info_ff_df['SAMPLE_AGE'].astype(float)
    seq_pat_info_ff_df['HEIGHT'] = seq_pat_info_ff_df['HEIGHT'].astype(float)
    seq_pat_info_ff_df['WEIGHT'] = seq_pat_info_ff_df['WEIGHT'].astype(float)

    seq_pat_info_ff_df.to_csv(out_fname, sep='\t', index=False)
    return seq_pat_info_ff_df


def get_tn_info(reload=False):
    """
    get negative samples: ie tn
    :param reload:
    :return:
    """

    data_dir_conf = get_config()['data_dir']
    data_in_root = data_dir_conf['data_root_dir']

    out_fname = os.path.join(data_in_root, 'tn.million.all.info.list')
    if reload:
        if not os.path.exists(out_fname):
            sys.exit("Error: {} not found.".format(out_fname))

        tn_df = pd.read_csv(out_fname, sep='\t', dtype=str)
        tn_df['SAMPLE_AGE'] = tn_df['SAMPLE_AGE'].astype(float)
        tn_df['HEIGHT'] = tn_df['HEIGHT'].astype(float)
        tn_df['WEIGHT'] = tn_df['WEIGHT'].astype(float)
        tn_df['FF'] = tn_df['FF'].astype(float)

        return tn_df

    # get tp, fp, fn
    seq_types = ['tp', 'fp', 'fn']
    chrs = ['21', '18', '13']

    tmp_samples = []
    seq_fnames = [seq_type + '.' + chr + '.million.bampath.list' for seq_type in seq_types for chr in chrs]
    for seq_fname in seq_fnames:
        seq_full_name = os.path.join(data_in_root, seq_fname)
        if os.path.exists(seq_full_name):
            seq_df = pd.read_csv(seq_full_name, sep='\t')
            tmp_samples.extend(seq_df['Sample_id'].values)
    tp_fp_fn_samples = np.unique(np.array(tmp_samples))
    logging.info('total unique number of tp+fp+fn={}'.format(len(tp_fp_fn_samples)))

    mil_sample_fname = data_dir_conf['million_sample_lst']
    milli_info_fname = data_dir_conf['million_info_fname']
    ff_fname = data_dir_conf['ff_info']

    if not os.path.exists(mil_sample_fname):
        sys.exit("Error: {} not found.".format(mil_sample_fname))
    if not os.path.exists(milli_info_fname):
        sys.exit("Error: {} not found.".format(milli_info_fname))
    if not os.path.exists(ff_fname):
        sys.exit("Error: {} not found.".format(ff_fname))

    milli_df = pd.read_csv(mil_sample_fname, sep='\t')
    tn_seq_milli_df = milli_df[~milli_df['Sample_id'].isin(tp_fp_fn_samples)]

    # load pat_info_df
    # Columns have mixed types. Specify dtype option on import or set low_memory=False.
    # So we simply read as str
    pat_info_df = pd.read_csv(milli_info_fname, sep='\t', dtype=str)
    tn_pat_info_df = pat_info_df[~pat_info_df['SAMPLE_NUM'].isin(tp_fp_fn_samples)]

    # load fetal fraction
    ff_df_t = pd.read_csv(ff_fname, usecols=[0, 1])
    ff_df_t.columns = ['FF_Sample_id', 'FF']
    # same sample have multiple ff
    ff_df = ff_df_t.groupby('FF_Sample_id')["FF"].agg('max').reset_index()
    tn_ff_df = ff_df[~ff_df['FF_Sample_id'].isin(tp_fp_fn_samples)]

    # load patient info
    # make sure height and weight is not null
    tn_pat_df = tn_pat_info_df[(~tn_pat_info_df['HEIGHT'].isnull()) &
                             (~tn_pat_info_df['WEIGHT'].isnull())]

    # further make sure there has ff values
    tn_pat_ff_df = pd.merge(tn_pat_df, tn_ff_df, how='inner', left_on=['SAMPLE_NUM'], right_on=['FF_Sample_id'])

    tn_df = pd.merge(tn_seq_milli_df, tn_pat_ff_df, how='inner', left_on=['Sample_id'], right_on=['SAMPLE_NUM'])
    # some age have chinese character, like '26岁'
    # extract number
    age = tn_df['SAMPLE_AGE'].str.extract('(\d+)', expand=False)
    tn_df['SAMPLE_AGE'] = age.astype(float)

    # height like 19+4
    height = tn_df['HEIGHT'].str.extract('(\d+)', expand=False)
    tn_df['HEIGHT'] = height.astype(float)

    weight = tn_df['WEIGHT'].str.extract('(\d+)', expand=False)
    tn_df['WEIGHT'] = weight.astype(float)

    tn_df.to_csv(out_fname, sep='\t', index=False)
    return tn_df






