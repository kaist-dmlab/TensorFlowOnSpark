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

# TF function to run on Spark

from db_parser import DBParser
from db_manager import DBManager

def read_query(query_path):
    file = open(query_path, "r")
    query = file.read()
    file.close()
    return query

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=1)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
    parser.add_argument("--sql_query_path", help="select sql_query_path")
    parser.add_argument("--model_dir", help="path to save checkpoint", default="mnist_model")
    parser.add_argument("--export_dir", help="path to export saved_model", default="mnist_export")
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

    args = parser.parse_args()
    print("args:", args)
    sql_query = read_query(args.sql_query_path)

    dbp = DBParser()
    dbm = DBManager()

    # Parse
    sql_query, hdfs_path, task, task_path = dbp.parse(sql_query)

    # Select with Classification
    dbm.select_sql(args, hdfs_path+"/csv", sql_query, task, task_path)