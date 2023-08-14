# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class ANLI(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "anli"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }

    def get_train_test_lines(self, dataset):
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        train_lines = map_hf_dataset_to_list(dataset, "train_r1")
        test_lines = map_hf_dataset_to_list(dataset, "dev_r1")
        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("premise: " + datapoint["premise"].replace("\n", " ") + " [SEP] hypothesis: " + datapoint["hypothesis"].replace("\n", " "), self.label[datapoint["label"]]))

        return lines

    def load_dataset(self):
        return datasets.load_dataset('anli')

def main():
    dataset = ANLI()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
