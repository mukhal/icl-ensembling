# Exploring Demonstration Ensembling for In-context Learning

This is code for our [paper](https://arxiv.org/abs/2308.08780) Exploring Demonstration Ensembling for In-context Learning
published at the workshop on understanding foundation models at ICLR 2023. 

The code here is largely based on this [repository](https://github.com/Alrope123/rethinking-demonstrations), so many thanks to the authors of that amazing repo! 

<img width="1408" alt="image" src="https://github.com/mukhal/icl-ensembling/assets/5109053/1c0ea10b-66d5-4a3b-9f85-f0416f242bfc">



### 1. Preprocess and prepare data for 6 and 10-shot. 
```
cd preprocess/
python _build_gym.py --build --n_proc=40 --do_test --test_k 6
python _build_gym.py --build --n_proc=40 --do_test --test_k 10
```
### 2. Run few-shot learning 

#### Weighted ensembling no clustering 
```bash
DATASET_NAME=glue-sst2 
N_DEMOS=10 # number of demos
NBUCKETS=5 # number of demo buckets
ENS_METHOD=max # max, MoE, or PoE
WEIGHTED=true # whether to weigh buckets by similarity with a test example 
CLUSTER=false # whether to use cluster the demos into bins
BSZ=64 # batch size 

python test.py --max_length 1024 --model $MODEL --use_demonstrations \
               --out_results_dir $OUTDIR \
               --test_batch_size $BSZ \
               --k $N_DEMOS \
               --n_ensemble $NBINS \
               --ensemble_method $ENS_METHOD \
               --dist_ensemble $WEIGHTED \
               --cluster_demos true \
```

#### Weighted ensembling with similar-together clustering 
```bash
DATASET_NAME=glue-sst2 
N_DEMOS=10 # number of demos
NBUCKETS=5 # number of demo buckets
ENS_METHOD=max # max, MoE, or PoE
WEIGHTED=true # whether to weigh buckets by similarity with a test example 
CLUSTER=true # whether to use cluster the demos into bins
BSZ=64 # batch size 

python test.py --max_length 1024 --model $MODEL --use_demonstrations \
               --out_results_dir $OUTDIR \
               --test_batch_size $BSZ \
               --k $N_DEMOS \
               --n_ensemble $NBINS \
               --ensemble_method $ENS_METHOD \
               --dist_ensemble $WEIGHTED \
               --cluster_demos true \
               --dissimilar_together false
```
For diverse clustering, pass `--dissimilar_together true`. For similar-together clustering, pass `--dissimilar_together false` 
