# Demonstration Ensembling


```
cd preprocess/
python _build_gym.py --build --n_proc=40 --do_test --test_k 16
```

#### 3. Update configs in `scripts/experiment_gptj_16.sh` (cuda devices, DATADIR env variable, output directory, etc.)


#### 4. run experiment

```
scripts/experiment_gptj_16.sh
```

