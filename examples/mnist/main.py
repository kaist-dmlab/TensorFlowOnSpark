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

from examples.mnist.db_parser import DBParser
from examples.mnist.db_manager import DBManager

# from DBManager import DBManager
class DBManager():
    def __init__(self):
        print("==========Start DBManager==========")

    def to_csv(self, example):
        return ','.join([str(i) for i in example.reshape(-1)])

    def parse_header(self, ln):
        vec = [str(x) for x in ln.split(',')]
        return vec

    def parse_xy(self, ln):
        vec = [float(x) for x in ln.split(',')]
        return (vec[1:], vec[0])

    def insert(self, in_local_path, out_hdfs_path, num_partitions=10):
        # FOR TESTING
        input = np.random.rand(101,3)
        input[0] = np.array([100,200,300])

        #input = np.loadtxt(in_local_path, delimiter=',', dtype=str)
        header = input[0]
        data = input[1:]

        sc = SparkContext(conf=SparkConf().setAppName("data_setup"))

        # Split Data into Partitions
        #header_rdd = sc.parallelize(np.array([header for i in range(0,num_partitions)]), num_partitions).cache()
        header_rdd = sc.parallelize(header, 1).cache()
        data_rdd = sc.parallelize(data, num_partitions).cache()

        # Save Data Partitions
        header_rdd.map(self.to_csv).saveAsTextFile(out_hdfs_path+"/csv/header")
        data_rdd.map(self.to_csv).saveAsTextFile(out_hdfs_path+"/csv/data")

    def select(self, col_names=None, filepath=None, sampler=None, sampling_rate=1, task="classfication"):
        sc = SparkContext(conf=SparkConf().setAppName("select"))

        # Columns Selection
        if col_names == None:
            print("============No Column Names Specified============")
            data_rdd = sc.textFile(filepath).map(self.parse_xy)
        else:
            header_rdd = sc.textFile(filepath+"/header").map(self.parse_header)
            header_list = sum(header_rdd.collect(), [])
            #print("header list: ", header_list)

            columns_ind = [header_list.index(col) for col in col_names]  # [Y, X1, X2, ...]
            X_col_ind = columns_ind[1:]
            Y_col_ind = columns_ind[0]
            #print("columns_index: ", columns_ind)

            # map function에 인자 전달??? 따로 빼고싶다,,
            data_rdd = sc.textFile(filepath+"/data").map(lambda vec: ([[float(x) for x in vec.split(',')][i] for i in X_col_ind], [float(x) for x in vec.split(',')][Y_col_ind]))

        if sampler == None:
            print("============No Sampling Method Specified============")
        elif sampler == "random":
            data_rdd = data_rdd.sample(False, sampling_rate)  # 정확하지가 않네..?

        print("data: ", data_rdd.collect()[0])
        print("len(data): ", len(data_rdd.collect()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_partitions", help="Number of output partitions", type=int, default=10)
    parser.add_argument("--hdfs_path", help="HDFS directory to save examples in parallelized format", default="data/mnist")

    args = parser.parse_args()
    print("args:", args)

    dbm = DBManager()

    ### DO INSERT ###
    #insert parameter
    infile_path = "data/sample/csv/infile.csv"
    outfile_path = args.hdfs_path
    num_partitions = 1

    dbm.insert(infile_path, outfile_path, num_partitions)


    ### DO SELECT ###
    #select parameter
    col_names = ['100.0', '200.0']
    data_file_path = args.hdfs_path+"/csv"
    sampler = "random"
    sampling_rate = 0.5
    task = "classification"

    #dbm.select(None, data_file_path, sampler, sampling_rate, task)
    #dbm.select(col_names, data_file_path, sampler, sampling_rate, task)