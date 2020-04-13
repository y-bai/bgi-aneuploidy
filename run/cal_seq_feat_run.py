#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: cal_seq_feat_run.py
    Description:
    
Created by YongBai on 2020/3/16 11:57 PM.
"""
import os
import pandas as pd
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    in_all_list = '/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/all_trainsample_chunk_1.cram.list'
    chr = '21'
    chr_len = 46709983

    cmd_str = """
    echo "python cal_seq_feat_main.py -p {0} -b {1} -o {2} -c {3} -l {4}" > {5}
    qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=80g,p=8 {5}
    """

    seq_out_root_dir = '/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/corrgc_bams'

    all_df = pd.read_csv(in_all_list, sep='\t')
    n_samples = len(all_df)
    n_submit = 0
    for row_idx, row in all_df.iterrows():
        sample_id = row['Sample_id']
        cram_full_fname = row['Cram_full_fname']

        sample_dir = os.path.join(seq_out_root_dir, sample_id)
        if not os.path.isdir(sample_dir):
            sys.exit('Error: {} not exist.'.format(sample_dir))

        cram_base_name = os.path.basename(cram_full_fname).rsplit('.', 1)[0]

        in_correctgc_fname = os.path.join(sample_dir, cram_base_name + '.gccorrect.chr' + chr + '.bam')
        if not os.path.exists(in_correctgc_fname):
            sys.exit('Error: {} not found.'.format(in_correctgc_fname))

        in_pileup_fname = os.path.join(sample_dir, cram_base_name + '.chr' + chr + '.pileup')
        if not os.path.exists(in_pileup_fname):
            sys.exit('Error: {} not found.'.format(in_pileup_fname))

        out_fname = os.path.join(sample_dir, cram_base_name + '.chr' + chr + '.features')

        if os.path.exists(out_fname) and os.path.getsize(out_fname) == 0:
            os.remove(out_fname)

        if os.path.exists(out_fname):
            logger.info('Warning: {} exist, skip...'.format(sample_id))
            # os.remove(out_fname)
            continue

        i_cmd_str = cmd_str.format(in_pileup_fname,
                                   in_correctgc_fname,
                                   out_fname,
                                   chr,
                                   chr_len,
                                   'run_feature_' + sample_id + '.sh')
        os.system(i_cmd_str)
        #
        logging.info('sumbmit {}/{} job for {}'.format(row_idx + 1, n_samples, sample_id))

        n_submit += 1
        # if row_idx >= 0:
        #     break
    print('# of submit {}'.format(n_submit))
