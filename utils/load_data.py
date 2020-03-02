#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: load_data.py
    Description:
    
Created by YongBai on 2020/3/2 1:21 PM.
"""
import os
import glob
import logging
import pandas as pd
from .config import get_config


def get_pos(chr='21', reload=False):
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
            tmp_df = pd.read_csv(fname, sep='\t', header=None)
            logging.info(fname)
            logging.info('# of samples of chr {}: {}'.format(chr, tmp_df.shape))
            if tmp_df.shape[0] > 0:
                tmp_sample_ids = tmp_df.iloc[:, 0].values
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













