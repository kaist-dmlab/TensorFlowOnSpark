import numpy as np
import tensorflow as tf
from pyspark.context import SparkContext
from pyspark.conf import SparkConf

from examples.mnist.db_parser import DBParser

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
        
    def select_and_train(self):
        pass

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
