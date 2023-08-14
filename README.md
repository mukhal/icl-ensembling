# Exploring Demonstration Ensembling for In-context Learning

This is code for our paper Exploring Demonstration Ensembling for In-context Learning
published at the workshop on understanding foundation models at ICLR 2023. 

The code here is largely based on this [repository](https://github.com/Alrope123/rethinking-demonstrations), so many thanks to the authors of that amazing repo! 


##### 1. Preprocess and prepare data. 
```
cd preprocess/
python _build_gym.py --build --n_proc=40 --do_test --test_k 16
```
##### 2. Run few-shot experiment with ensembling 
DATASET_NAME=glue-sst2 
N_DEMOS=10 # number of demos
NBUCKETS=5 # number of demo buckets
ENS_METHOD=max # max, MoE, or PoE
WEIGHTED=true # whether to weigh buckets by similarity with test example 
CLUSTER=false # whether to use cluster the demos into bins 

python test.py --max_length 1024 --model $MODEL --use_demonstrations \
               --out_results_dir $OUTDIR --test_batch_size $BSZ \
               --k $N_DEMOS \
               --n_ensemble $NBINS \
               --ensemble_method $ENS_METHOD \
               --dist_ensemble $WEIGHTED \
               --cluster_demos true \
               --dissimilar_together false true
