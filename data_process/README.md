# NIPT data process 
The process is based on [deeptools](https://deeptools.readthedocs.io/en/develop/content/list_of_tools.html).

## Compute GC bias 

we call `correctGCBias` from deeptools

###  Arguments for `correctGCBias`
* `--effectiveGenomeSize`: As we use GRCh38 as reference, thus the  `Effective size` is 2913022398 according to https://deeptools.readthedocs.io/en/latest/content/feature/effectiveGenomeSize.html.
* `--genome` or `-g`: function of `correctGCBias` uses genome in two bit format. so we have to convert the reference fasta files into 2bit file
  * download `faToTwoBit` from http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/faToTwoBit.
    ```shell script
    wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/faToTwoBit
    chmod +x faToTwoBit
    faToTwoBit hg38.fa hg38.2bit
    ```
  *
  
### download `faToTwoBit` from 
 