import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from tensorflowonspark import TFCluster


class DBManager():
    """DB Manager"""
    
    def __init__(self):
        print("==========Start DBManager==========")

    def main_fun(args, ctx):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import numpy as np
        import tensorflow as tf
        from tensorflowonspark import compat, TFNode

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

        def build_and_compile_cnn_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                metrics=['accuracy'])
            return model

        # single node
        # single_worker_model = build_and_compile_cnn_model()
        # single_worker_model.fit(x=train_datasets, epochs=3)

        tf_feed = TFNode.DataFeed(ctx.mgr, False)

        def rdd_generator():
            while not tf_feed.should_stop():
                batch = tf_feed.next_batch(1)
                if len(batch) > 0:
                    example = batch[0]
                    image = np.array(example[0]).astype(np.float32) / 255.0
                    image = np.reshape(image, (28, 28, 1))
                    label = np.array(example[1]).astype(np.float32)
                    label = np.reshape(label, (1,))
                    yield (image, label)
                else:
                    return

        ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32),
                                            (tf.TensorShape([28, 28, 1]), tf.TensorShape([1])))
        ds = ds.batch(args.batch_size)

        # this fails
        # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=args.model_dir)]
        tf.io.gfile.makedirs(args.model_dir)
        filepath = args.model_dir + "/weights-{epoch:04d}"
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True)]

        with strategy.scope():
            multi_worker_model = build_and_compile_cnn_model()

        # Note: MultiWorkerMirroredStrategy (CollectiveAllReduceStrategy) is synchronous,
        # so we need to ensure that all workers complete training before any of them run out of data from the RDD.
        # And given that Spark RDD partitions (and partition sizes) can be non-evenly divisible by num_workers,
        # we'll just stop training at 90% of the total expected number of steps.
        steps_per_epoch = 60000 / args.batch_size
        steps_per_epoch_per_worker = steps_per_epoch / ctx.num_workers
        max_steps_per_worker = steps_per_epoch_per_worker * 0.9

        multi_worker_model.fit(x=ds, epochs=args.epochs, steps_per_epoch=max_steps_per_worker, callbacks=callbacks)

        from tensorflow_estimator.python.estimator.export import export_lib
        export_dir = export_lib.get_timestamped_export_dir(args.export_dir)
        compat.export_saved_model(multi_worker_model, export_dir, ctx.job_name == 'chief')

        # terminating feed tells spark to skip processing further partitions
        tf_feed.terminate()

    def to_csv(self, example):
        return ','.join([str(i) for i in example.reshape(-1)])

    def to_csv_mnist(self, example): #only for MNIST TEST2
        return str(example['label']) + ',' + ','.join([str(i) for i in example['image'].reshape(784)])

    def parse(self, ln):
        vec = [str(x) for x in ln.split(',')]
        return vec

    def parse_xy(self, ln):
        vec = [float(x) for x in ln.split(',')]
        return (vec[1:], vec[0])

    def insert(self, in_local_path, out_hdfs_path, num_partitions=10):
        '''
        # FOR TESTING 1
        input = np.random.rand(101,3)
        input[0] = np.array([100,200,300])
        input[1] = np.array([10000, 20000, 30000])

        # FOR TESTING 2
        mnist, info = tfds.load('mnist', with_info=True)
        print("info: ", info.as_json)
        mnist_train = tfds.as_numpy(mnist['train'])
        mnist_test = tfds.as_numpy(mnist['test'])
        '''
        time_logs = np.zeros(5)
        time0 = time.time()

        print("1. Before read local input csv file")
        time_logs[0] = float(time.time() - time0)

        input = np.loadtxt(in_local_path, delimiter=',', dtype=str)
        header = input[0] #np.array(['Y']+['X'+str(i) for i in range(1, 785)]) #for TEST2
        data = input[1:] #mnist_train

        print("2. Before start spark clusters")
        time_logs[1] = float(time.time() - time0)

        sc = SparkContext(conf=SparkConf().setAppName("data_setup"))

        print("3. Before parallerlize")
        time_logs[2] = float(time.time() - time0)

        # Split Data into Partitions
        header_rdd = sc.parallelize(header, 1).cache()
        data_rdd = sc.parallelize(data, num_partitions).cache()

        print("4. Before save data on hdfs")
        time_logs[3] = float(time.time() - time0)

        # Save Data Partitions
        header_rdd.map(self.to_csv).saveAsTextFile(out_hdfs_path+"/csv/header")
        data_rdd.map(self.to_csv).saveAsTextFile(out_hdfs_path+"/csv/data")
        #data_rdd.map(self.to_csv_mnist).saveAsTextFile(out_hdfs_path + "/csv/data") #for TEST2

        print("5. After save data on hdfs")
        time_logs[4] = float(time.time() - time0)

        print("============Finish Insertion============")
        print("Time logs: ", time_logs)

    def select(self, col_names=None, filepath=None, sampler=None, sampling_rate=1, task="classfication"):
        sc = SparkContext(conf=SparkConf().setAppName("select"))

        # Columns Selection
        if col_names == None:
            print("============No Column Names Specified============")
            data_rdd = sc.textFile(filepath).map(self.parse_xy)
        else:
            header_rdd = sc.textFile(filepath+"/header").map(self.parse)
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

    def select_sql(self, args, infile_path, sql_query, task="classification", task_path="main_func.py"):
        sc = SparkContext(conf=SparkConf().setAppName("select_sql"))
        spark = SparkSession(sc)

        time_logs = np.zeros(5)
        time0 = time.time()

        # Read Header
        header_rdd = sc.textFile(infile_path + "/header").map(self.parse)
        header_list = sum(header_rdd.collect(), [])  #Should not have .
        #header_list = ["A", "B", "C"]  # For Test1
        #header_list = ['Y'] + ['X' + str(i) for i in range(1, 785)]  # For Test2
        #print("*header_list: ", *header_list)

        print("1. Before read hdfs files")
        time_logs[0] = float(time.time() - time0)

        # 1. Read Data
        data_rdd = sc.textFile(infile_path + "/data").map(self.parse)
        data_df = data_rdd.toDF(header_list)
        data_df.createOrReplaceTempView("temp")
        print("data initial: ", data_df.collect()[0])

        print("2. Before Do Spark_SQL")
        time_logs[1] = float(time.time() - time0)

        # 2. Do SQL
        data_df2 = spark.sql(sql_query)
        #print("data after sql: ", data_df2.collect()[0])

        print("3. Before Convert DataFrame into RDD([X1,X2,...], Y) ")
        time_logs[2] = float(time.time() - time0)

        # 3. Preprocessing DF into RDD([X1,X2,...], Y)
        data_rdd3 = data_df2.rdd
        if task == "classification":
            data_rdd3 = data_rdd3.map(lambda vec: (vec[1:], vec[0]))
        #print("data after preprocessing: ", data_rdd3.collect()[0])
        print("============Finish Selection============")

        print("4. Before Training")
        time_logs[3] = float(time.time() - time0)

        # 4. Train by TFoS
        print("============Start Training============")
        #self.load_main_func(task_path) #run main_func
        #from main_func import main_fun
        cluster = TFCluster.run(sc, self.main_fun, args, args.cluster_size, num_ps=0, tensorboard=args.tensorboard,
                                input_mode=TFCluster.InputMode.SPARK, master_node='chief')
        cluster.train(data_rdd3, args.epochs)

        print("4. After Training")
        time_logs[4] = float(time.time() - time0)

        cluster.shutdown()
        print("============Finish Training============")
        print("Time logs: ", time_logs)
