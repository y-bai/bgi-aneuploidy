# Predict Chromosome Aneuploidy

## Data Description
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
表示NIPT检测结果是FN的样本。也就是说文件夹`million.Insurance.database`下面都是recheck过的正样本（三倍体），而文件`million.Negative.recheck.with.bampath.list`
包含T21, T18, T13的所有NIPT检测是FP的样本（也就是真实是阴性即整倍体的样本）。

## Data Preparation
1. 使用`./data_preprocess/data_prepare.py`是处理文件夹`million.Insurance.database`内的`TP`和`FN`样本。

    调用代码：
    ```python
    from data_preprocess import get_seq_lst
    get_seq_lst(reload=False, seq_type='fn', chr='18', in_fname='million.Positive.False.18.recheck.with.bampath.list')
    ```
    
    这个方法是从`in_fname`得到`Sample_id`，然后在完整的百万测序文件列表中抽取对应`Sample_id`的测序文件列表，这样保证了所有输出结果的格式（如文件表头）一致。

    注：这个方法只处理`tp`和`fn`的样本。方法的输出结果为（13号染色体没有`FN`）：
    ```
    tp.21.million.bampath.list
    tp.18.million.bampath.list
    tp.13.million.bampath.list 
    fn.21.million.bampath.list
    fn.18.million.bampath.list
    ```

2. 同样使用`./data_preprocess/data_prepare.py`处理文件`million.Negative.recheck.with.bampath.list`得到`FP`样本.

    ```python
    from data_preprocess import get_fp_seq_lst
    get_fp_seq_lst(chr='21', reload=False)
    ```
    这个方法是从`million.Negative.recheck.with.bampath.list`得到给定`chr`的`Sample_id`，然后在完整的百万测序文件列表中抽取对应`Sample_id`的测序文件列表，这样保证了所有输出结果的格式（如文件表头）一致。

    注：这个方法只处理`FP`样本， 方法的输出结果为：
    ```
    fp.21.million.bampath.list 
    fp.18.million.bampath.list 
    fp.13.million.bampath.list 
    ```

3. 使用`./data_preprocess/data_prepare.py`得到所有`TP`,`FP`,`FN`的完整数据。完成数据包括：测序结果文件，样本的meta数据（如身高体重等）和FF信息（`lims.FMD.csv`包含FF浓度）。
    ```python
    from data_preprocess import get_sample_info
    get_sample_info(reload=False, seq_type='tp', chr='21')
    ```
    输出结果为：
    ```
    tp.21.million.all.info.list
    tp.18.million.all.info.list
    tp.13.million.all.info.list
    fn.21.million.all.info.list
    fn.18.million.all.info.list
    fp.21.million.all.info.list
    fp.18.million.all.info.list
    fp.13.million.all.info.list

    ```
4. 在百万列表中抽取`TN`样本集
    ```python
    from data_preprocess import get_tn_info
    get_tn_info(reload=False)
    ```
    
    这个方法在百万列表中抽取`TN`样本，同时这些抽取的`TN`样本需要保证其FF、HEIGHT、WEIGHT都不为空。
    这个方法的结果为文件`tn.million.all.info.list`.
    
    另外，为了构造训练集，我们在`tn.million.all.info.list`随机抽取1600个样本来构造训练集。方法为：
    ```python
    from data_preprocess import get_tn_sample_4train
    get_tn_sample_4train(reload=False, n_samples=1600)
    ```
    这个方法的结果为文件`tn.million.4train.info.list`.
    

## GC 校正
1. 生成所有样本对应的测序结果文件列表

    这里调用`deepTools`的GC校正方法。所以可以将`TP`,`FP`,`FN`和用于训练的1600个`TN`合并成一个文件提交任务，但是由于qsub限制了一次提交的任务数量（2000），所以需要将合并后的文件进行切分，方法如下：

    ```python
    from data_preprocess import get_unique_cram_lst
    get_unique_cram_lst(chunk_size=1900)
    ```

    这个方法在`seq_prepare.py`中，该方法将合并后的文件切分成两个文件：
    ```
    all_trainsample_chunk_0.cram.list
    all_trainsample_chunk_1.cram.list
    ```
    这两个文件只包含了`sample_id`和对应测序结果的`.cram`文件的绝对路径。
    
2. GC校正
    有了上面的两个文件就可以调用`deepTools`的相应方法进行GC校正。
    ```python
    from data_preprocess import compute_gc_bias, correct_gc_bias
    compute_gc_bias('/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/all_trainsample_chunk_0.cram.list')
    # compute_gc_bias('/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/all_trainsample_chunk_1.cram.list')
   
    # compute_gc_bias 计算完成后调用correct_gc_bias进行GC校正
    correct_gc_bias('/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/all_trainsample_chunk_0.cram.list', chr='21')
    ```
    以上两个方法在`seq_prepare.py`中。目前已经完成了所有21，18和13的所有样本的GC校正。
    
3. 计算RC, Base quality和Map quality
    
    首先在`seq_feature.py`中调用`samtools mpileup`方法生成`.pileup`文件：
    
    ```python
    from data_preprocess import gen_pileup_batch
    gen_pileup_batch('/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/all_trainsample_chunk_1.cram.list', chr='18')
    ```
    **目前完成了21和18号染色体的pileup文件，13号染色体还没有生成。**
    
    进入`run`目录，在命令行运行如下命令
    ```shell script
    nohup python cal_seq_feat_run.py &
    ```
    在每个以样本标号为目录的文件夹下生成`*.features`的seq feature.
    
    **目前完成了21号染色体所有样本（包括TP, FP, TN, FN）的所有序列特征。**

4. 构建训练集和测试集
    为了方便模型训练，这里需要使用TP和TN样本构建训练集和测试集。**这里只完成21号染色体的训练集构建。**
    
    ```python
    from data_preprocess import get_train_test_samples
    get_train_test_samples(chr='21', f_type='tp', reload=False)
    # get_train_test_samples(chr='21', f_type='tn', reload=False)
    ```
    生成的文件为：
    ```
    tp.21.million.train.info.list
    tp.21.million.test.info.list
    tn.million.4train.train.info.list
    tn.million.4train.test.info.list
    ```
       
## 深度神经网络训练并生成seq数据的特征
* 使用训练集训练网络，使用测试集测试网络
    ```python
    from model import train_run
    # 模型最终参数
    param = {}
    param['chr'] = '21'
    param['epochs'] = 100
    param['batch'] = 64
    param['lr'] = 1e-2
    param['filters'] = 32
    param['kernel_size'] = 8
    param['drop'] = 0.5
    param['l2r'] = 1e-4
    param['n_gpu'] = 1
    
    train_run('/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/model', **param)
    
    # 使用测试集评估模型
    # param['include_top'] = True
    # param.pop('n_gpu', None)
    # pos_pred, neg_pred = predict_run('/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/model', **param)
    # print(pos_pred)
    # print(neg_pred)
    ```
    模型存储在目录`/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/model`下。
    
    ---
    
    为了训练最终网络模型，我们将训练集和测试集合并后训练最终网络模型。
    ```python
    from model import retrain_final_net
    # 模型最终参数
    param = {}
    param['chr'] = '21'
    param['epochs'] = 100
    param['batch'] = 64
    param['lr'] = 1e-2
    param['filters'] = 32
    param['kernel_size'] = 8
    param['drop'] = 0.5
    param['l2r'] = 1e-4
    param['n_gpu'] = 1
    param['include_top'] = True
  
    param.pop('include_top', None)
    retrain_final_net('/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/model', **param)
  
    ```
    这个方法的结果是得到最终网络模型，同时生成scaler的模型文件`scaler_final.pkl`，用于scale输入测序数据。
    
* 生成测序数据特征
    ```python
    from model import cal_seq_feat_net
    
    cal_seq_feat_net('/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/aneuploidy_data/model', 
      chr='21', seq_type='tn',filters=32, kernel_size=8, drop=0.5, l2r=1e-4, final_model=True)
    ```
    **这个方法可以改变`chr='21', seq_type='tn'`这两个参数生成对应的sequence的特征。**
    
    这个方法生成的文件在`model/feat_gen`路径下，文件名如下：
    ```
    chr21_seqnet_final_tp.seqfeature
    chr21_seqnet_final_tn.seqfeature
    ```
  
## 合并seq特征和meta信息用于训练集成模型
```python
    from ensemble import read_features
    read_features(reload=False, chr='21')
```
这个方法生成的文件在`model/feat_gen`路径下，文件名如下：
```
chr21_normalized_test.allfeature
chr21_normalized_train.allfeature
```
同时这个方法生成了`imputer`处理缺失值，和`scaler`处理meta数据的normalization. 文件在`model`目录下，文件名为：
```
pheno-imputor.pkl
pheno-scaler.pkl
```

## 训练集成模型
使用上面生成的`chr21_normalized_test.allfeature`和`chr21_normalized_train.allfeature`训练集成模型。

```python
from ensemble import ens_model_train_run
ens_model_train_run(chr='21', kfold=5)
```

## 预测
见`analysis/f21_ens_predict.ipynb`

---

# 问题记录
上面的在进行test预测的时候，AUC=1。但是预测FP和FN时的准确度不高，很显然是overfitting了。

重新修改模型：

 