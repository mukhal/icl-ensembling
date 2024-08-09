# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, random
import csv
import json
import string
import numpy as np
import pickle as pkl
import math
import torch

from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from itertools import permutations

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from embedding_distance import SimilarityModel, ClusteringModel


class MetaICLData(object):

    def __init__(self, logger=None, tokenizer=None, method="channel", use_demonstrations=True, k=16,
                 max_length=1024, max_length_per_example=450,
                 do_tensorize=False, tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1,
                 do_ensemble=False, n_ensemble=1, truncate_demos=False, dist_ensemble=False, cluster_demos=False,
                 dissimilar_together=False, demo_shuffle_seed=None, subset_demos=None, sort_by_similarity=False):
                
        
        self.logger = logger
        self.tokenizer = tokenizer
        self.method = method
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example
        self.truncate_demos = truncate_demos

        self.do_tensorize = do_tensorize
        self.tensorize_dir = tensorize_dir
        self.n_process = n_process
        self.n_gpu = n_gpu
        self.local_rank = local_rank
        self.do_ensemble = do_ensemble
        self.dist_ensemble = dist_ensemble
        self.n_ensemble = n_ensemble
        self.cluster_demos = cluster_demos
        self.dissimilar_together = dissimilar_together
        self.demo_shuffle_seed = demo_shuffle_seed
        self.sort_by_similarity = sort_by_similarity
        self.subset_demos = subset_demos
        
        assert k % n_ensemble == 0, "# demos must be divisible by number of ensembles"
        if sort_by_similarity or demo_shuffle_seed:
            assert bool(sort_by_similarity) ^ bool(demo_shuffle_seed), "Must specify either sort_by_similarity or demo_shuffle_seed"

        self.tensorized_inputs = None
        self.metadata = None

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        device = 0 #"cuda:" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[-1].strip()
        if dist_ensemble or sort_by_similarity:
            logger.info("Loading similarity model for distance ensembling...")
            self.similarity_model = SimilarityModel(device=device)

        if sort_by_similarity and self.n_ensemble > 1:
            raise NotImplementedError("Currently only support concat when sorting by similarity")

        if subset_demos is not None:
            assert subset_demos <= k, "subset_demos must be <= k"

        if cluster_demos:
            logger.info("Loading clutsering model...")
            self.clustering_model = ClusteringModel(device=device)

        
    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[MetaICL Data]: method=%d, "
        if self.use_demonstrations:
            if self.do_ensemble:
                text += "demo ensembling - %d demos x %d ensembles, " % (self.k // self.n_ensemble, self.n_ensemble)
            else:
                text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def get_dataloader(self, batch_size, is_training):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        
        shape = inputs["input_ids"].shape
        self.logger.info(shape)
        for v in inputs.values():
            assert v.shape==shape
        if "labels" in inputs:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["labels"])
        else:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        if is_training:
            sampler=RandomSampler(dataset)
        else:
            sampler=SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def evaluate(self, predictions, groundtruths, is_classification):
        assert len(predictions)==len(self.metadata)
        accs = []
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for prediction, groundtruth in zip(predictions, groundtruths):
            prediction = prediction.strip()
            groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth)==list else groundtruth.strip()
            is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
            accs.append(is_correct)
            if is_classification:
                recalls[groundtruth].append(is_correct)
                precisions[prediction].append(is_correct)

        if not is_classification:
            return np.mean(accs)

        f1s = []
        for key in recalls:
            precision = np.mean(precisions[key]) if key in precisions else 1.0
            recall = np.mean(recalls[key])
            if precision+recall==0:
                f1s.append(0)
            else:
                f1s.append(2*precision*recall / (precision+recall))

        return np.mean(f1s)

    def _prepro_each_datapoint(self, dp, is_first=True, is_training=False, for_demonstrations=False,
                               add_newlines=True):
        
        dp = dp.copy()
        if add_newlines:
            no_label = np.all([option=="" for option in dp["options"]])
            no_input = dp["input"]==""
            if self.method=="direct":
                if not is_first:
                    if no_input:
                        dp["input"] = "\n\n" + dp["input"]
                    else:
                        dp["input"] = "\n\n\n" + dp["input"]
                if not no_label:
                    dp["output"] = "\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n" + opt for opt in dp["options"]]
            elif self.method=="channel":
                if not is_first:
                    dp["output"] = "\n\n\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n\n\n" + opt for opt in dp["options"]]
                if not no_input:
                    if not no_label:
                        dp["input"] = "\n" + dp["input"]
            else:
                raise NotImplementedError()
        else:
            if not is_first:
                if self.method=="direct":
                    dp["input"] = " " + dp["input"]
                elif self.method=="channel":
                    dp["output"] = " " + dp["output"]
                    if "options" in dp:
                        dp["options"] = [" "+opt for opt in dp["options"]]
                else:
                    raise NotImplementedError()
            if self.method=="direct":
                dp["output"] = " " + dp["output"]
                if "options" in dp:
                    dp["options"] = [" " + opt for opt in dp["options"]]
            elif self.method=="channel":
                dp["input"] = " " + dp["input"]
            else:
                raise NotImplementedError()

        input_tokens = self.tokenizer(dp["input"])["input_ids"]
        if is_training or for_demonstrations:
            output_tokens = self.tokenizer(dp["output"])["input_ids"]

            if "task" in dp:
                if (dp["task"].startswith("inst:piqa") or dp["task"].startswith("inst:yahoo_answers_topics")) and \
                        len(input_tokens)+len(output_tokens)+2>self.max_length_per_example:
                    input_tokens = input_tokens[:self.max_length_per_example // 2]
                    output_tokens = output_tokens[:self.max_length_per_example // 2 - 2]

                elif len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                    if dp["task"].startswith("inst:") and len(input_tokens)<len(output_tokens):
                        output_tokens = output_tokens[:self.max_length_per_example - 2 - len(input_tokens)]
                    else:
                        input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, \
                (dp.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)

            if self.method=="direct":
                return input_tokens, output_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()

        else:
            assert len(dp["options"])>=2, dp
            assert dp["output"] in dp["options"]
            option_tokens = [self.tokenizer(option)["input_ids"] for option in dp["options"]]
            option_length = np.max([len(option) for option in option_tokens])

            if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
                input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]

            input_tokens = [input_tokens for _ in option_tokens]
            output_tokens = option_tokens
            option_tokens = [dp["options"].index(dp["output"])]

            if self.method=="direct":
                return input_tokens, output_tokens, option_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens, option_tokens
            else:
                raise NotImplementedError()

    def _tensorize_for_training(self, train_data):
        for dp in train_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        n_answers = []

        if self.use_demonstrations:
            first_tokenized = []
            nonfirst_tokenized = []

            for dp in train_data:
                first_tokenized.append(self._prepro_each_datapoint(
                    dp, is_first=True, is_training=True))
                nonfirst_tokenized.append(self._prepro_each_datapoint(
                    dp, is_first=False, is_training=True))

            N=1

            def _draw_random(tot, n, exclude_indices):
                r = np.random.choice([i for i in range(tot) if i not in exclude_indices])
                if n==1:
                    return [r]
                return [r] + _draw_random(tot, n-1, exclude_indices | set([r]))

            for dp_idx, dp in enumerate(train_data):
                for _ in range(N):
                    demo_indices = _draw_random(len(train_data), self.k, set([dp_idx]))
                    inputs = []
                    for demo_idx, index in enumerate(demo_indices):
                        if demo_idx==0:
                            inputs += first_tokenized[index][0] + first_tokenized[index][1]
                        else:
                            inputs += nonfirst_tokenized[index][0] + nonfirst_tokenized[index][1]
                        assert index!=dp_idx
                    inputs += nonfirst_tokenized[dp_idx][0]
                    outputs = nonfirst_tokenized[dp_idx][1]

                    encoded = prepro_sentence_pair_single(
                        inputs, outputs, self.max_length, bos_token_id, eos_token_id,
                        allow_truncation=True)

                    input_ids.append(encoded[0])
                    attention_mask.append(encoded[1])
                    token_type_ids.append(encoded[2])

        else:
            for dp in train_data:
                inputs, outputs = self._prepro_each_datapoint(
                    dp, is_first=True, is_training=True)

                encoded = prepro_sentence_pair_single(
                    inputs, outputs, self.max_length, bos_token_id, eos_token_id)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        return dict(input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask),
                    token_type_ids=torch.LongTensor(token_type_ids))


    def _tensorize_for_training_with_random_english_words(self, train_data):
        for dp in train_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        n_answers = []

        from english_words import english_words_set
        english_words_set = sorted(english_words_set)

        if self.use_demonstrations:
            N=1
            def _draw_random(tot, n, exclude_indices):
                r = np.random.choice([i for i in range(tot) if i not in exclude_indices])
                if n==1:
                    return [r]
                return [r] + _draw_random(tot, n-1, exclude_indices | set([r]))

            for dp_idx, dp in enumerate(train_data):
                for _ in range(N):
                    demo_indices = _draw_random(len(train_data), self.k, set([dp_idx]))
                    inputs = []

                    # create a mapping
                    mapping = {option: np.random.choice(english_words_set) for option in dp["options"]}

                    for demo_idx, index in enumerate(demo_indices):
                        curr_demo_dp = train_data[index].copy()
                        curr_demo_dp["output"] = mapping[curr_demo_dp["output"]]
                        curr_demo_dp["options"] = [mapping[o] for o in curr_demo_dp["options"]]
                        inputs_, outputs_ = self._prepro_each_datapoint(curr_demo_dp, is_first=demo_idx==0, is_training=True)
                        inputs += inputs_ + outputs_
                        assert index!=dp_idx

                    curr_dp = dp.copy()
                    curr_dp["output"] = mapping[curr_dp["output"]]
                    curr_dp["options"] = [mapping[o] for o in curr_dp["options"]]
                    inputs_, outputs = self._prepro_each_datapoint(curr_dp, is_first=False, is_training=True)
                    inputs += inputs_
                    encoded = prepro_sentence_pair_single(
                        inputs, outputs, self.max_length, bos_token_id, eos_token_id,
                        allow_truncation=True)
                    input_ids.append(encoded[0])
                    attention_mask.append(encoded[1])
                    token_type_ids.append(encoded[2])
        else:
            for dp in train_data:
                # create a mapping
                mapping = {option: np.random.choice(english_words_set) for option in dp["options"]}
                dp["output"] = mapping[dp["output"]]
                dp["options"] = [mapping[o] for o in dp["options"]]
                inputs, outputs = self._prepro_each_datapoint(
                    dp, is_first=True, is_training=True)
                encoded = prepro_sentence_pair_single(
                    inputs, outputs, self.max_length, bos_token_id, eos_token_id)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        return dict(input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask),
                    token_type_ids=torch.LongTensor(token_type_ids))

    def tensorize(self, _train_data, _test_data, options=None,
                  add_newlines=True):

        if options is not None:
            assert np.all([dp["output"] in options for dp in _train_data])
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp)==str
                _test_data[i] = {"input": dp, "options": options}

        if self.demo_shuffle_seed is not None:
            idx = np.arange(len(_train_data))
            ## get all permutations with itertools.permutations
            all_permutations = list(permutations(idx))
            ## shuffle the permutations
            np.random.seed(self.demo_shuffle_seed)
            new_idx = np.random.choice(len(all_permutations))
            _train_data = [_train_data[i] for i in all_permutations[new_idx]]

        if self.subset_demos is not None:
            _train_data = _train_data[:self.subset_demos]
        
        train_data, test_data = [], []
        if self.use_demonstrations:
            for dp in _train_data:
                assert type(dp)==dict, ("Each example should be a dictionary", dp)
                assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                train_data.append(dp.copy())
            
            ensemble_idx_to_train_idx = None
            if self.n_ensemble > 1:
                ensemble_idx_to_train_idx = {}
                demo_per_ensemble = int(len(_train_data) / self.n_ensemble)
                
                if not self.cluster_demos:
                    for i, _ in enumerate(_train_data):
                            ensemble_idx = i // demo_per_ensemble
                            if ensemble_idx not in ensemble_idx_to_train_idx:
                                ensemble_idx_to_train_idx[ensemble_idx] = []
                            ensemble_idx_to_train_idx[ensemble_idx].append(i)
                else: 
                    ## clustering demos 
                    examples = [d['input'] for d in _train_data]

                    cluster_labels = self.clustering_model.cluster_examples(examples, k=self.n_ensemble, dissimilar_together=self.dissimilar_together)
                    for ex_idx, ens_idx in enumerate(cluster_labels):
                        if ens_idx not in ensemble_idx_to_train_idx:
                            ensemble_idx_to_train_idx[ens_idx] = []
                        ensemble_idx_to_train_idx[ens_idx].append(ex_idx)          

        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"])==list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0] # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        if self.use_demonstrations:
            if self.subset_demos is None:
                assert len(train_data)==self.k
            else: 
                assert len(train_data)==self.subset_demos
            
            demonstrations = []

            if self.n_ensemble == 1:
                if self.sort_by_similarity:
                ## create a demonstrations list for EVERY test example that represents the demos sorted by similarity to that test example
                    print("********* SORTING DEMOS BY SIMILARITY *********")
                    for test_dp in test_data:
                        demos = []
                        for train_dp in train_data:
                            demos.append((train_dp, self.similarity_model.get_similarity(test_dp["input"], train_dp["input"])))
                        demos = sorted(demos, key=lambda x: x[1], reverse=True)
                        demos = [d[0] for d in demos]
                        demonstrations.append([])
                        for i, dp in enumerate(demos):
                            input_, output_ = self._prepro_each_datapoint(
                            dp, is_first=i==0, for_demonstrations=True,
                            add_newlines=add_newlines)
                            demonstrations[-1] += input_ + output_

                    assert len(demonstrations) == len(test_data)

                else:
                    for i, dp in enumerate(train_data):
                        input_, output_ = self._prepro_each_datapoint(
                            dp, is_first=i==0, for_demonstrations=True,
                            add_newlines=add_newlines)
                        demonstrations += input_ + output_
            else:
                demonstrations = [[] for _ in range(len(ensemble_idx_to_train_idx))]
                for ens_idx, train_idx in ensemble_idx_to_train_idx.items():
                    for i in train_idx:
                        dp = train_data[i]
                        input_, output_ = self._prepro_each_datapoint(
                            dp, is_first=i==0, for_demonstrations=True,
                            add_newlines=add_newlines)
                        demonstrations[ens_idx] += input_ + output_

        indices_counter = 0
        for ex_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            
            assert len(inputs) == len(dp["options"])
            indices = [[i] for i in range(indices_counter, indices_counter + len(inputs))] # store the indices of the ensembles for each example
            indices_counter += len(inputs)

            assert len(indices) == len(inputs)
            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    if self.n_ensemble == 1:
                        if self.sort_by_similarity:
                            assert len(demonstrations) == len(test_data)
                            assert isinstance(demonstrations[ex_idx], list)
                            demos = demonstrations[ex_idx]
                            inputs_ = demos + inputs_
                        else: 
                            inputs_ = demonstrations + inputs_
                    else:
                        inputs_ = [demonstrations[ens_idx] + inputs_ for ens_idx in ensemble_idx_to_train_idx]
                    
                if self.n_ensemble == 1:
                    encoded = prepro_sentence_pair_single(
                        inputs_, outputs_, self.max_length, bos_token_id, eos_token_id,
                        allow_truncation=self.truncate_demos)
                    
                    assert len(encoded[0]) <= self.max_length
                    input_ids.append(encoded[0])
                    attention_mask.append(encoded[1])
                    token_type_ids.append(encoded[2])
                
                else:
                    encoded = [prepro_sentence_pair_single(
                        in_, outputs_, self.max_length, bos_token_id, eos_token_id,
                        allow_truncation=self.truncate_demos) for in_ in inputs_]

                    assert len(encoded) == self.n_ensemble
                    for i, enc in enumerate(encoded):
                        input_ids.append(enc[0])
                        attention_mask.append(enc[1])
                        token_type_ids.append(enc[2])
            
            if self.n_ensemble > 1 and self.dist_ensemble:
                metadata[-1]["similarity_with_demos"] = []
                ## compute distance from each ensemble                
                for ens_idx, demos_idx in ensemble_idx_to_train_idx.items():
                    demo_str = [train_data[i]["input"] for i in demos_idx]
                    sims = [self.similarity_model.get_similarity(dp["input"], d) for d in demo_str]
                    sim = np.mean(sims) # average similarity with demos in the ensemble
                    metadata[-1]["similarity_with_demos"].append(sim)
            else:
                metadata[-1]["similarity_with_demos"] = None
        
        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                    attention_mask=torch.LongTensor(attention_mask),
                                    token_type_ids=torch.LongTensor(token_type_ids))
        #else:
        #    self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids).view(-1, self.n_ensemble, self.max_length),
        #                                attention_mask=torch.LongTensor(attention_mask).view(-1, self.n_ensemble, self.max_length),
        #                                token_type_ids=torch.LongTensor(token_type_ids).view(-1, self.n_ensemble, self.max_length))
        
        self.metadata = metadata

    def tensorize_for_training(self, train_data, keyword, seed, use_random_english_words=False):
        assert self.tensorize_dir is not None

        if not os.path.exists(self.tensorize_dir):
            os.makedirs(self.tensorize_dir)

        method_name = self.method + "-demon" if self.use_demonstrations else self.method
        k_name = "%d-%d" % (len(train_data), self.k) if self.use_demonstrations else len(train_data)
        length_name = "%d-%d" % (self.max_length, self.max_length_per_example) if self.use_demonstrations else self.max_length
        postfix = "-randomEnglish" if use_random_english_words else ""

        tensorize_path = os.path.join(self.tensorize_dir,
                                      "{}_{}_k={}_seed={}_length={}{}-rank=%d.pkl".format(
                                          keyword, method_name, k_name, seed, length_name,
                                          postfix))

        if self.local_rank==-1:
            self.logger.info(tensorize_path)
        else:
            self.logger.info(tensorize_path % self.local_rank)
        all_tensorize_paths = [tensorize_path % i for i in range(self.n_gpu)]

        if not self.do_tensorize:
            if not np.all([os.path.exists(_path) for _path in all_tensorize_paths]):
                self.logger.info("Tensorization was not done. Run with `--do_tensorize` without distributed mode"
                            "and then run training command again")
                raise NotImplementedError()

            if self.local_rank==-1:
                inputs = defaultdict(list)
                for i in range(self.n_gpu):
                    with open(tensorize_path % i, "rb") as f:
                        curr_inputs = pkl.load(f)
                    for k, v in curr_inputs.items():
                        inputs[k] += v
            else:
                assert 0<=self.local_rank<self.n_gpu
                with open(tensorize_path % self.local_rank, "rb") as f:
                    inputs = pkl.load(f)

            self.tensorized_inputs = inputs
            return

        assert self.local_rank==-1
        if any([os.path.exists(_path) for _path in all_tensorize_paths]):
            self.logger.info("tensorize file already exists...")
            return

        unique_task_names = set([dp["task"] for dp in train_data])
        sharded_inputs = []
        if self.use_demonstrations or (len(unique_task_names)>200 and len(train_data)>=1638400):
            tot = 0
            for i, curr_train_task in enumerate(unique_task_names):
                curr_train_data = [dp for dp in train_data if dp["task"]==curr_train_task]
                tot += len(curr_train_data)
                if self.use_demonstrations and len(unique_task_names)>200 and len(train_data)>=1638400:
                    # data is too huge; sampling 10% of the data
                    self.logger.info("Sampling training data from %d to %d", len(curr_train_data), len(curr_train_data)//10)
                    indices = np.random.permutation(range(len(curr_train_data)))[:len(curr_train_data)//10]
                    curr_train_data = [curr_train_data[i] for i in indices]
                elif len(unique_task_names)>200 and len(train_data)>=1638400:
                    # data is too huge; sampling 50% of the data
                    self.logger.info("Sampling training data from %d to %d", len(curr_train_data), len(curr_train_data)//2)
                    indices = np.random.permutation(range(len(curr_train_data)))[:len(curr_train_data)//2]
                    curr_train_data = [curr_train_data[i] for i in indices]
                sharded_inputs.append(curr_train_data)
            assert len(train_data)==tot
        else:
            n_per_shard = math.ceil(len(train_data) / self.n_process)
            for i in range(self.n_process):
                sharded_inputs.append(train_data[i*n_per_shard:(i+1)*n_per_shard])

        inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        _tensorize_for_training = self._tensorize_for_training_with_random_english_words \
            if use_random_english_words else self._tensorize_for_training
        if self.n_process==1:
            for in_ in sharded_inputs:
                out = _tensorize_for_training(in_)
                for key in ["input_ids", "attention_mask", "token_type_ids"]:
                    inputs[key] += out[key].numpy().tolist()
        else:
            with Pool(self.n_process) as p:
                for out in p.imap_unordered(_tensorize_for_training, sharded_inputs):
                    for key in ["input_ids", "attention_mask", "token_type_ids"]:
                        inputs[key] += out[key].numpy().tolist()

        N = len(inputs["input_ids"])
        indices = np.random.permutation(range(N))
        for k, v in inputs.items():
            inputs[k] = np.array(v)[indices]
        n_per_shard = math.ceil(N / self.n_gpu)

        for i, _path in enumerate(all_tensorize_paths):
            start = i*n_per_shard
            end = (i+1)*n_per_shard
            curr_inputs = {k:v[start:end].tolist() for k, v in inputs.items()}
            with open(_path, "wb") as f:
                pkl.dump(curr_inputs, f)
            self.logger.info("Preprocessing done for i=%d" % i)

        self.logger.info("Finish saving preprocessed data ...")

    def print_tensorized_example(self, return_string=False):
        assert self.tensorized_inputs is not None

        text = "Checking the first example..." if self.n_ensemble == 1 else "Checking {} ensembles for the first example".format(self.n_ensemble)

        for idx in range(self.n_ensemble):
            input_ids = self.tensorized_inputs["input_ids"][idx]
            token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
            if type(input_ids)!=list:
                input_ids = input_ids.numpy().tolist()
            
            if type(token_type_ids)!=list:
                token_type_ids = token_type_ids.numpy().tolist()
            text += "\n[{}] Input:\n".format(idx)
            text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
            #text += "\nOutput:\n"
            #text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])

        if return_string:
            return text

        if self.local_rank<=0:
            self.logger.info(text)

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                bos_token_id, eos_token_id,
                                allow_truncation=False):

    #if bos_token_id is not None:
    #    ids1 = [bos_token_id] + ids1
    #if eos_token_id is not None:
    #    ids2 = ids2 + [eos_token_id]
    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    return input_ids, attention_mask, token_type_ids

def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}

