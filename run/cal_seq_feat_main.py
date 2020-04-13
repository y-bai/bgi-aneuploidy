#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: cal_seq_feat_main.py
    Description:
    
Created by YongBai on 2020/3/16 11:21 PM.
"""

import sys
import argparse
import logging
sys.path.append('..')
from data_preprocess import cal_seq_feature

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):

    in_pileup_fname = args.in_pileup_fname
    in_gc_correct_bam_fname = args.in_gc_correct_bam_fname
    out_fname = args.out_fname

    param = {}
    param['chr'] = args.chr
    param['ref_chr_len'] = args.chr_ref_len
    param['bin_size'] = 1000000
    param['step_size'] = 500000
    param['n_proc'] = 8

    cal_seq_feature(in_pileup_fname, in_gc_correct_bam_fname, out_fname, **param)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculating sequence feature')

    parser.add_argument(
        "-p",
        "--in_pileup_fname",
        type=str)

    parser.add_argument(
        "-b",
        "--in_gc_correct_bam_fname",
        type=str)

    parser.add_argument(
        "-o",
        "--out_fname",
        type=str)

    parser.add_argument(
        "-c",
        "--chr",
        type=str)

    parser.add_argument(
        "-l",
        "--chr_ref_len",
        type=int)

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)
