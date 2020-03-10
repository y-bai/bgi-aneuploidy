#!/usr/bin/env sh

# =====================================================
# Description: seq_process.sh
#
# =====================================================
#
# Created by YongBai on 2020/3/4 1:38 PM.

hg38_ref="/zfssz6/ST_MCHRI/BIGDATA/database/BGI-seq500_OSS_download/human_reference/hg38/Homo_sapiens_assembly38.fasta"

test_cram="/zfssz6/ST_MCHRI/BIGDATA/Million_seq500/subsample_database/CL100093941_L01_12/CL100093941_L01_12.sorted.rmdup.realign.BQSR.cram"

output_bam="/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/bams/CL100093941_L01_12.sorted.rmdup.realign.BQSR.bam"
output_bai="/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/bams/CL100093941_L01_12.sorted.rmdup.realign.BQSR.bai"
output_bastat="/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/bams/CL100093941_L01_12.sorted.rmdup.realign.BQSR.castat"


#echo "samtools view -T ${hg38_ref} -@ 20 --write-index ${test_cram} -b -o ${output_bam}##idx##${output_bai}" > run_script.sh
#qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=40g,p=1 run_script.sh
echo "samtools stats -T ${hg38_ref} ${test_cram} -@ 20 > ${output_bastat}">run_script.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=40g,p=1 run_script.sh