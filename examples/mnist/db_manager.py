import numpy as np
import pandas as pd
import tensorflow as tf
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import re


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
        input = np.random.rand(101, 3)
        input[0] = np.array([100, 200, 300])

        # input = np.loadtxt(in_local_path, delimiter=',', dtype=str)
        header = input[0]
        data = input[1:]

        sc = SparkContext(conf=SparkConf().setAppName("data_setup"))

        # Split Data into Partitions
        # header_rdd = sc.parallelize(np.array([header for i in range(0,num_partitions)]), num_partitions).cache()
        header_rdd = sc.parallelize(header, 1).cache()
        data_rdd = sc.parallelize(data, num_partitions).cache()

        # Save Data Partitions
        header_rdd.map(self.to_csv).saveAsTextFile(out_hdfs_path + "/csv/header")
        data_rdd.map(self.to_csv).saveAsTextFile(out_hdfs_path + "/csv/data")

    def select_and_train(self):
        pass

    def select(self, data_col='*', label_col=None, source=None, where_cond=None, output_root=None):  # NEW VERSION
        # Data Load
        if source == None:
            return "NO SOURCE ERROR"

        ## only one file (if you need to deal with multiple folder, you should change the code below)
        ## in this case, we consider only one local data
        data = pd.read_csv(source)
        orignal_list = list(data.columns)
        result_data = pd.DataFrame()
        col_list = list()

        # SELECT clause
        ## data_col
        for tmp_name in data_col:
            if tmp_name == "*":
                col_list += orignal_list
            else:
                if (" as " in tmp_name) or (" AS " in tmp_name):  # exist `as`
                    if (" as " in tmp_name):
                        name_before, name_after = tmp_name.split(" as ")
                    else:
                        name_before, name_after = tmp_name.split(" AS ")

                    data.eval(name_after + "=" + name_before, inplace=True)
                    col_list += [name_after]
                else:  # no exist `as`
                    if tmp_name in list(data.columns):
                        col_list += [tmp_name]
                    else:  # need to use assign
                        new_name = re.sub('[^0-9a-zA-Zㄱ-힗_]', '', tmp_name)
                        data.eval(new_name + "=" + tmp_name, inplace=True)
                        col_list += [new_name]

        ## label_col
        if label_col not in col_list:
            col_list += [label_col]

        ## real selecting
        result_data = data[col_list].copy()
        # del [data]  #free

        # WHERE clause
        ## pre-processing for is null and query
        if where_cond != None:
            where_cond = where_cond.replace(" AND ", " and ")
            where_cond = where_cond.replace(" OR ", " or ")
            where_cond = where_cond.replace("=", "==")
            where_cond = where_cond.replace("<==", "<=")  # exception
            where_cond = where_cond.replace(">==", ">=")  # exception

            result_data.query(where_cond, inplace=True)

        # save
        output_name = str(np.random.rand(1)[0]).split(".")[1]  # random number
        if output_root != None:
            output_path = output_root + "/" + output_name + ".csv"
        else:
            output_path = output_name + ".csv"  # making at current working directory

        result_data.to_csv(output_path, index=False)
        return output_path