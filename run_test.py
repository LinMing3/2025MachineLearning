# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "D:/LibFewShot/results/ft_fda"
VAR_DICT = {
    "test_epoch": 1,
    "device_ids": "0",
    "n_gpu": 1,
    "test_episode": 300,
    "episode_size": 1,
    "data_root": "D:/dataset/libfewshot_datasets/aircraft",
    # "test_way": 5,
    # "test_shot": 5,
    # "test_query": 10,
    # "augment_times": 1,
    # "augment_times_query": 1,
    # "way_num": 5,
    # "shot_num": 5,
    # "query_num": 10,
}


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
