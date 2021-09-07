# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse 
import numpy as np
import time

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from tensorflowonspark import TFCluster

import tensorflow as tf
import tensorflow_datasets as tfds

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

from db_parser import DBParser
from db_manager import DBManager

def read_query(query_path):
    file = open(query_path, "r")
    query = file.read()
    file.close()
    return query

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sql_query_path", help="SQL Query", type=str)

    args = parser.parse_args()
    print("args:", args)
    sql_query = read_query(args.sql_query_path)

    dbp = DBParser()
    dbm = DBManager()

    # Parse
    in_local_path, out_hdfs_path, num_partitions = dbp.parse(sql_query)

    # Insert
    dbm.insert(in_local_path, out_hdfs_path, num_partitions)
