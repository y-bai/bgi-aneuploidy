#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: prepare_data.py
    Description:
    
Created by YongBai on 2020/3/2 1:21 PM.
"""
import os
import glob
import logging
import pandas as pd
from .config import get_config


def get_pos(chr='21', reload=False):
    """
    including NIPT true positive and false negative
    :param chr:
    :param reload:
    :return:
    """
    # get million data list

    conf = get_config()
    _input = conf['data_dir']
    data_root_dir = _input['data_root_dir']
    out_pos_fname = os.path.join(data_root_dir, 'million_postive_' + chr + '.list')

    if reload:
        re_df = pd.read_csv(out_pos_fname, sep='\t')
        logging.info('# of total samples of {}: {}'.format(chr, re_df.shape))
        return re_df
    else:
        input_pos_dir = os.path.join(data_root_dir, _input['input_pos_dir'])
        milli_df = pd.read_csv(_input['million_sample_lst'], sep='\t')

        logging.info('# of million samples: {}'.format(milli_df.shape))
        # get 'chr'  list
        chr_pos_files = glob.glob(os.path.join(input_pos_dir,
                                               'million.Positive.*.' + chr + '.recheck.with.bampath.list'))
        re_df = None
        for fname in chr_pos_files:
            tmp_df = pd.read_csv(fname, sep='\t', header=None, usecols=[0], names=['Sample_id'])
            logging.info(fname)
            logging.info('# of samples of chr {}: {}'.format(chr, tmp_df.shape))
            if tmp_df.shape[0] > 0:
                tmp_sample_ids = tmp_df['Sample_id'].values
                tmp_df = milli_df[milli_df['Sample_id'].isin(tmp_sample_ids)]
                tmp_df['DSource'] = os.path.basename(fname)
                if re_df is None:
                    re_df = tmp_df
                else:
                    re_df = pd.concat([re_df, tmp_df], axis=0, sort=False)

        logging.info('# of total pos samples of chr {}: {}'.format(chr, re_df.shape))
        if os.path.exists(out_pos_fname):
            os.remove(out_pos_fname)

        re_df.to_csv(out_pos_fname, sep='\t', index=False)
        logging.info('Done, results saved at: {}'.format(out_pos_fname))
        return re_df


def get_fp(chr='21', reload=False):
    """
    get NIPT false positive samples
    :param chr:
    :param reload:
    :return:
    """
    conf = get_config()
    _input = conf['data_dir']
    data_root_dir = _input['data_root_dir']

    out_pos_fname = os.path.join(data_root_dir, 'million_nipt_false_postive_' + chr + '.list')
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













