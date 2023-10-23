# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import tensorflow as tf
import tensorflow_datasets as tfds

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from db_parser import DBParser
from db_manager import DBManager

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_partitions", help="Number of output partitions", type=int, default=1)
    parser.add_argument("--infile_path", default="data/your_data/train_data_name.csv")
    parser.add_argument("--outfile_path", default="data/mnist")
    
    args = parser.parse_args()
    print("args:", args)

    dbm = DBManager()
    
    ### DO INSERT ###
    dbm.insert(infile_path, outfile_path, num_partitions)