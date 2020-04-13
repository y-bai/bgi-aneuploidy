#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: data_split.py
    Description:
    
Created by YongBai on 2020/3/18 3:30 PM.
"""
import os
from sklearn.model_selection import train_test_split
from .data_prepare import _load_samples, get_tn_sample_4train, get_sample_info
import sys
sys.path.append('..')
from utils import get_config


def get_train_test_samples(chr='21', f_type='tp', reload=False):

    data_out_root_dir = get_config()['data_dir']['data_root_dir']

    if not os.path.isdir(data_out_root_dir):
        sys.exit('Error: dir {} not found.'.format(data_out_root_dir))

    if reload:
        if f_type == 'tn':
            tn_train_out_fname = os.path.join(data_out_root_dir, 'tn.million.4train.train.info.list')
            tn_test_out_fname = os.path.join(data_out_root_dir, 'tn.million.4train.test.info.list')

            tn_train_df = _load_samples(tn_train_out_fname)
            tn_test_df = _load_samples(tn_test_out_fname)

            return tn_train_df, tn_test_df
        else:
            tp_train_out_fname = os.path.join(data_out_root_dir, f_type + '.' + chr + '.million.train.info.list')
            tp_test_out_fname = os.path.join(data_out_root_dir, f_type + '.' + chr + '.million.test.info.list')

            tp_train_df = _load_samples(tp_train_out_fname)
            tp_test_df = _load_samples(tp_test_out_fname)

            return tp_train_df, tp_test_df

    # load tn samples
    if f_type == 'tn':
        tn_df = get_tn_sample_4train(reload=True)
        tn_train_df, tn_test_df = train_test_split(tn_df, test_size=0.1, random_state=11)

        tn_train_out_fname = os.path.join(data_out_root_dir,  'tn.million.4train.train.info.list')
        tn_test_out_fname = os.path.join(data_out_root_dir, 'tn.million.4train.test.info.list')

        tn_train_df.to_csv(tn_train_out_fname, sep='\t', index=False)
        tn_test_df.to_csv(tn_test_out_fname, sep='\t', index=False)

        return tn_train_df, tn_test_df
    else:
        # load tp samples
        tp_df = get_sample_info(reload=True, seq_type=f_type, chr=chr)

        tp_train_df, tp_test_df = train_test_split(tp_df, test_size=0.1, random_state=11)
        tp_train_out_fname = os.path.join(data_out_root_dir, f_type + '.' + chr + '.million.train.info.list')
        tp_test_out_fname = os.path.join(data_out_root_dir, f_type + '.' + chr + '.million.test.info.list')

        tp_train_df.to_csv(tp_train_out_fname, sep='\t', index=False)
        tp_test_df.to_csv(tp_test_out_fname, sep='\t', index=False)

        return tp_train_df, tp_test_df

