#!/usr/bin/env sh

# =====================================================
# Description: run.sh.sh
#
# =====================================================
#
# Created by YongBai on 2020/3/6 1:49 PM.

test_cram="/zfssz6/ST_MCHRI/BIGDATA/Million_seq500/subsample_database/CL100093941_L01_12/CL100093941_L01_12.sorted.rmdup.realign.BQSR.cram"
ref_2bit="/zfssz6/ST_MCHRI/BIGDATA/database/BGI-seq500_OSS_download/human_reference/hg38/Homo_sapiens_assembly38.2bit"
out_freq="/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_predict/freq_gc.txt"
out_plot="/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_predict/gc.png"

echo "python cal_gc_bias.py -b ${test_cram} \
--effectiveGenomeSize 2913022398 \
-g ${ref_2bit} \
-p 20 \
-freq ${out_freq} -l 180 --regionSize 50000 --biasPlot ${out_plot}" > run_script.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=40g,p=1 run_script.sh