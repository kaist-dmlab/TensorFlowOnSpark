<!--
Copyright 2019 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
-->
# TensorSparkML
> _TensorSparkML brings scalable in-memory machine learning to TensorFlow on Apache Spark clusters._

[![Build Status](https://travis-ci.org/yahoo/TensorFlowOnSpark.svg?branch=master)](https://travis-ci.org/yahoo/TensorFlowOnSpark) [![PyPI version](https://badge.fury.io/py/tensorflowonspark.svg)](https://badge.fury.io/py/tensorflowonspark)

By combining salient features from the [TensorFlow](https://www.tensorflow.org) deep learning framework with [Apache Spark](http://spark.apache.org) and [Apache Hadoop](http://hadoop.apache.org), TensorFlowOnSpark enables distributed
deep learning on a cluster of GPU and CPU servers.

It enables both distributed TensorFlow training and
inferencing on Spark clusters, with a goal to minimize the amount
of code changes required to run existing TensorFlow programs on a
shared grid.  Its Spark-compatible API helps manage the TensorFlow
cluster with the following steps:

1. **Startup** - launches the Tensorflow main function on the executors, along with listeners for data/control messages.
1. **Data ingestion**
   - **InputMode.TENSORFLOW** - leverages TensorFlow's built-in APIs to read data files directly from HDFS.
   - **InputMode.SPARK** - sends Spark RDD data to the TensorFlow nodes via a `TFNode.DataFeed` class.  Note that we leverage the [Hadoop Input/Output Format](https://github.com/tensorflow/ecosystem/tree/master/hadoop) to access TFRecords on HDFS.
1. **Shutdown** - shuts down the Tensorflow workers and PS nodes on the executors.

## Background

TensorSparkML provides some important benefits over alternative deep learning solutions.
   * Easily migrate existing TensorFlow programs with <10 lines of code change.
   * Support all TensorFlow functionalities: synchronous/asynchronous training, model/data parallelism, inferencing and TensorBoard.
   * Server-to-server direct communication achieves faster learning when available.
   * Allow datasets on HDFS and other sources pushed by Spark or pulled by TensorFlow.
   * Easily integrate with your existing Spark data processing pipelines.
   * Easily deployed on cloud or on-premise and on CPUs or GPUs.

For distributed clusters, please see our [wiki site](../../wiki) for detailed documentation for specific environments, such as our getting started guides for [single-node Spark Standalone](https://github.com/kaist-dmlab/TensorSparkML/wiki/GetStarted_Standalone), [YARN clusters](../../wiki/GetStarted_YARN) and [AWS EC2](../../wiki/GetStarted_EC2).  Note: the Windows operating system is not currently supported due to [this issue](https://github.com/kaist-dmlab/TensorSparkML/issues/36).

## Usage

To use TensorSparkML with an existing TensorFlow application, you can follow our [Conversion Guide](../../wiki/Conversion-Guide) to describe the required changes.  Additionally, our [wiki site](../../wiki) has pointers to some presentations which provide an overview of the platform.

**Note: since TensorFlow 2.x breaks API compatibility with TensorFlow 1.x, the examples have been updated accordingly.  If you are using TensorFlow 1.x, you will need to checkout the `v1.4.4` tag for compatible examples and instructions.**

## License

The use and distribution terms for this software are covered by the Apache 2.0 license.
See [LICENSE](LICENSE) file for terms.
