# Predict Chromosome Aneuploidy

# Data Description
1. `million.Negative.recheck.with.bampath.list`

    这个数据集里面包含了假阳样本数据，也就是 NIPT 检测出来是阳性，但是最后在进行复检结果是阴性，即NIPT的结果为假阳的数据。
    
    复检过程是通过羊穿等进行。确定是否是正阳或假阳是通过保险赔付来确定的。真阳和假阳的保险赔付不同，此次来判断NIPT结果是否是真阳或假阳。
    
    数据格式
    ```
    sample_id1      46;XN   F/T/F/N   seq_lane_id      sample_id1     bam_related_files
    sample_id2      46;XN   T/F/F/N   seq_lane_id      sample_id2     bam_related_files
    ```
   关键是第三列，第一到第三字符分别表示13，18、21的 NIPT 检测结果，F表示对应的染色体在NIPT没有检出异常，T表示对应的染色体在NIPT检出了三倍体。
   第四个字符表示真值的结果，N表示真实情况下没有发生异常（也就是说NIPT是False Positive）
   
   如果第四个字符是P，表示真值是出现了对应染色体的三倍体。这个时候第二列会指明具体是那个染色体是三倍体。比如21号染色体是三倍体：
   ```
    sample_id1      47;XN;+21       F/F/T/P seq_lane_id      sample_id1      bam_related_files
    sample_id2      47;XN;+21       F/F/T/P seq_lane_id      sample_id1      bam_related_files
   ```
2. `million.Positive.True.21.recheck.with.bampath.list`表示NIPT检测是TP的样本，`million.Positive.False.18.recheck.with.bampath.list`
表示NIPT检测结果是FN的样本。
   
 