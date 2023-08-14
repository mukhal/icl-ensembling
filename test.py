# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from email.charset import add_charset
from operator import sub
import os
import argparse
import pickle as pkl
import random
from sklearn import cluster
import torch
import math
import json
import string
import logging
import numpy as np
import sys

from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import GPT2Tokenizer, AutoTokenizer

from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel

from utils.data import load_data
from sacred import Experiment

ex = Experiment()
from sacred.observers import MongoObserver, FileStorageObserver
from neptune.new.integrations.sacred import NeptuneObserver

NEPTUNE_API_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmODAzMTM3My1kN2E0LTQ1MGEtYTAzNS01M2M0ZDQyMmYxNTgifQ=="


@ex.main
def main(_run):
    logger = logging.getLogger(__name__)
    logger.info(args)
    
    assert (args.dataset is not None and args.task is None) or (args.dataset is None and args.task is not None)
    ex.logger = logger

    if args.model.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    elif "gpt-j" in args.model:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=os.environ["TRANSFORMERS_CACHE"])
    elif "llama" in args.model:
        tokenizer = AutoTokenizer.from_pretrained("models/llama-7b")
    else:
        raise NotImplementedError("model %s not supported" % args.model)
    add_newlines = True

    ### checkpoint ...
    if not args.do_zeroshot:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            assert args.global_step is None
        else:
            assert args.global_step is not None
            checkpoint = os.path.join(args.out_dir, "model-{}.pt".format(args.global_step))
        assert os.path.exists(checkpoint)
    else:
        add_newlines = (not args.model.startswith("gpt2")) and (not "llama" in args.model)
        checkpoint = None
    
    metaicl_model = MetaICLModel(logger, args.out_dir, ensemble_method=args.ensemble_method)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data

    if args.truncate_demos:
        max_length_per_example =  1024 // (args.k + 1)
    else:
        ## use the maximum length in trainin data 
        config_split = "unseen_domain_test" if args.unseen_domain_only else "test"
        train_data = load_data(args.task, "train", args.k, seed=args.mseed.split(",")[0], config_split=config_split,
                               datasets=None if args.dataset is None else args.dataset.split(","))
        
        max_length_per_example = max([len(tokenizer.encode(x["input"])) for x in train_data])

    max_length = args.max_length

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(logger, tokenizer, args.method,args.use_demonstrations, args.k,
                               max_length, max_length_per_example, 
                               n_ensemble=args.n_ensemble, dist_ensemble=args.dist_ensemble,
                               cluster_demos = args.cluster_demos, dissimilar_together=args.dissimilar_together,
                               demo_shuffle_seed=args.demo_shuffle_seed, 
                               sort_by_similarity=args.sort_by_similarity,
                               subset_demos=args.subset_demos,
                               truncate_demos=args.truncate_demos)

    results = []
    errors = []
    seeds = args.mseed.split(",")
    config_split = "unseen_domain_test" if args.unseen_domain_only else "test"

    for seed in seeds:
        ### data ...
        train_data = load_data(args.task, "train", args.k, seed=seed, config_split=config_split,
                               datasets=None if args.dataset is None else args.dataset.split(","))
        dev_data = load_data(args.task, args.split, args.k, seed=seed, config_split=config_split,
                             datasets=None if args.dataset is None else args.dataset.split(","), is_null=args.is_null)

        if args.use_random_english_words:
            from english_words import english_words_set
            english_words_set = sorted(english_words_set)
            np.random.seed(int(seed))

        train_counter = Counter()
        dev_counter = Counter()
        for dp in train_data:
            train_counter[dp["task"]] += 1
        for dp in dev_data:
            dev_counter[dp["task"]] += 1
        for k, v in train_counter.items():
            logger.info("[Train] %s\t%d" % (k, v))
        for k, v in dev_counter.items():
            logger.info("[Dev] %s\t%d" % (k, v))
        
        logger.info("%s on %s (%d train, %d dev)" % (args.method, args.task, len(train_counter), len(dev_counter)))

        for test_task in dev_counter:
            curr_dev_data = [dp for dp in dev_data if dp["task"]==test_task]
            curr_train_data = [dp for dp in train_data if dp["task"]==test_task]
            assert len(curr_dev_data)>0
            assert not args.use_demonstrations or len(curr_train_data)==args.k, \
                    (args.use_demonstrations, len(curr_train_data), args.k)

            config_file = "config/tasks/{}.json".format(test_task)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = config["task_type"]=="classification"
            if is_classification:
                options = curr_dev_data[0]["options"]
                assert np.all([d["options"]==options for d in curr_dev_data])

            if args.use_random_english_words:
                # create a mapping
                options = curr_dev_data[0]["options"]
                mapping = {option: np.random.choice(english_words_set) for option in options}
                new_options = list(mapping.values())
                for dp_idx, dp in enumerate(curr_train_data):
                    assert dp["output"] in options, (dp, options)
                    curr_train_data[dp_idx]["output"] = mapping[dp["output"]]
                    curr_train_data[dp_idx]["options"] = new_options
                for dp_idx, dp in enumerate(curr_dev_data):
                    assert dp["output"] in options, (dp, options)
                    curr_dev_data[dp_idx]["output"] = mapping[dp["output"]]
                    curr_dev_data[dp_idx]["options"] = new_options
                
            result = run(logger, test_task, metaicl_data, metaicl_model,
                         curr_train_data, curr_dev_data, seed, checkpoint, is_classification, add_newlines, 
                         _run)

            if result is None:
                errors.append("%s/%s" % (test_task, seed))
            else:
                results.append(result)

    if args.is_null:
        return

    logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.task, len(results) // len(seeds), 100*np.mean(results)))
    _run.log_scalar("Macro-F1 ", np.mean(results) * 100)

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


def run(logger, task, metaicl_data, metaicl_model, train_data, dev_data, seed,
        checkpoint, is_classification, add_newlines, _run):

    if args.do_zeroshot:
        split_name = args.split
        if args.is_null:
            split_name += "-null"
        cache_path = os.path.join(args.out_dir,
                                  "{}-{}-{}{}{}{}{}{}{}.pkl".format(
                                      task,
                                      split_name,
                                      metaicl_data.method,
                                      "-k={}".format(args.k) if args.use_demonstrations else "",
                                      "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
                                      "-n_ens={}".format(args.n_ensemble) if args.n_ensemble > 1 else "",
                                      "-ens_method={}".format(args.ensemble_method) if args.n_ensemble > 1 else "",
                                      "" if add_newlines else "-no-newlines",
                                      "-randomEnglish" if args.use_random_english_words else ""))
    else:
        assert add_newlines
        cache_path = os.path.join(args.out_dir, "{}-{}-{}{}{}{}.pkl".format(
                        task,
                        args.split,
                        metaicl_data.method,
                        "-k={}".format(args.k) if args.use_demonstrations else "",
                        "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
                        "-randomEnglish" if args.use_random_english_words else ""
                      ))

    metaicl_data.tensorize(train_data, dev_data, add_newlines=add_newlines)
    metaicl_data.print_tensorized_example()
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")
    if args.use_calibration:
        prediction_path = prediction_path.replace(".txt", "-calibrated.txt")
    
    if metaicl_model.is_none():
        metaicl_model.load(checkpoint, gpt2=args.model)
        metaicl_model.cuda()
        metaicl_model.eval()

        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)
        with open(cache_path, "wb") as f:
            pkl.dump(losses, f)

    assert len(losses)==len(metaicl_data)


    if args.is_null:
        return None

    if args.use_calibration:
        assert args.do_zeroshot
        bias_path = cache_path.replace("/"+task+"-"+args.split, "/"+task+"-"+args.split+"-null")
        assert os.path.exists(bias_path), bias_path
        with open(bias_path, "rb") as f:
            bias_losses = pkl.load(f)

        losses = np.array(losses)
        bias_losses = np.array(bias_losses)
        assert losses.shape == bias_losses.shape
        losses -= bias_losses

    preds = metaicl_model.do_predict(metaicl_data, losses=losses, get_probs=True)
    predictions = preds['predictions']
    probs = preds['per_label_probs']
    per_ensemble_ll = preds['per_ensemble_ll']
    ### compute entropy of the probabilities for each example 
    entropies = [-np.sum(p * np.log(p)) for p in probs]
    ### compute the average entropy of the probabilities for each example
    avg_entropy = np.mean(entropies)
    logger.info("Average entropy of the probabilities: {}".format(avg_entropy))
    _run.log_scalar("AvgEnt", avg_entropy)

    groundtruths = [dp["output"] for dp in dev_data]
    #import ipdb; ipdb.set_trace()
    perf = metaicl_data.evaluate(predictions, groundtruths, is_classification)
    logger.info("Accuracy=%s" % perf)
    
    _run.log_scalar("Accuracy", perf * 100)
    #wandb.log({"Accuracy": perf * 100})

    info_to_save = []
    for dp, pred, prob, ens_ll in zip(dev_data, predictions, probs, per_ensemble_ll):
        dp.update({"prediction": pred, "prob": prob, "per_ensemble_ll": ens_ll,
        "demos": train_data})
        info_to_save.append(dp)

    ## writing the predictions to a file
    with open(cache_path + '_all', "wb") as f:
        pkl.dump(info_to_save, f)

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return perf

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_zeroshot", default=True, action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--unseen_domain_only", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--mseed", type=str, default="100")

    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")
    parser.add_argument("--out_dir", type=str, default="/tmp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--is_null", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--model", type=str, default="gpt-j")
    parser.add_argument("--n_ensemble", type=int, default=1, help="# of demo bins (ensembles). Set to -1 to have K ensembles")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--ensemble_method", type=str, default=None, choices=["poe", "moe", "max", "min", "most_similar", None])
    parser.add_argument("--dist_ensemble", default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help= "use distance from text example as weights in case of MoE or PoE")
    parser.add_argument("--cluster_demos", default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument("--dissimilar_together", default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="whether to group together demos from different clusters within a bin")
    parser.add_argument("--demo_shuffle_seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--sort_by_similarity", default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="whether to sort demos by similarity with the input")
    parser.add_argument("--subset_demos", type=int, default=None, help="subset size of demos")
    parser.add_argument("--mongo_db", type=str, default=None) #"my_db")
    parser.add_argument("--out_results_dir", type=str, default=None)
    parser.add_argument("--truncate_demos", action='store_true', default=False)
    

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)

    args_dict = vars(args)
    relevant_args = [arg for arg in args_dict if arg not in ["log_file", "method", "is_null", 
    "task", "use_calibration", "unseen_domain_only", "global_step", "checkpoint", "use_random_english_words", "out_dir", "do_zeroshot", "max_length", ]
    ]

    args_dict = {arg: args_dict[arg] for arg in relevant_args}
    ex.add_config(args_dict)

    if args.ensemble_method in ["most_similar"] and not args.dist_ensemble:
        print("need to set dist_ensemble when using most_similar ensembling")
        sys.exit(0)

    if args.cluster_demos and not args.dist_ensemble:
        print("need to set dist_ensemble when using cluster_demos")
        sys.exit(0)

    if not args.debug:
        if args.mongo_db is not None:
            ex.observers.append(MongoObserver(url='mongodb://localhost:27017/', db_name=args.mongo_db))
        if args.out_results_dir is not None:
            print("adding file storage observer")
            if not os.path.exists(args.out_results_dir):
                os.makedirs(args.out_results_dir)
            
            ex.observers.append(FileStorageObserver.create(args.out_results_dir))

    ex.run()

