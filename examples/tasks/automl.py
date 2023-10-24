def main_fun(args, ctx):
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = ""

  import numpy as np
  import tensorflow as tf

  from tensorflowonspark import compat, TFNode

  # import TimesNAS relevant packages
  from TimesNAS import searcher, search_configs, search_space
  from TimesNAS import experiments as exp_setup
  from TimesNAS.exp.exp_imputation import Exp_Imputation
  from TimesNAS.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
  from TimesNAS.exp.exp_anomaly_detection import Exp_Anomaly_Detection
  from TimesNAS.exp.exp_classification import Exp_Classification  

  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  def search_and_compile_model(task_name, data_name):
    assert task_name in ['forecasting', 'classification', 'anomaly_detection', 'imputation']

    # SETUP SEARCH CONFIGS
    search_configs.task_name = task_name
    search_configs.data_name = data_name
    Exp = {
      'forecasting': Exp_Long_Term_Forecast,
      'classification': Exp_Classification,
      'anomaly_detection': Exp_Anomaly_Detection,
      'imputation': Exp_Imputation
    }

    # RUN SEARCH (pytorch model -> onnx -> tf model)
    model = searcher.run(Exp, search_configs)

    # COMPILE MODEL
    # model.compile(loss=..., optimizer=..., metrics=[...])
    
    return model

  # single node
  #single_worker_model = build_and_compile_cnn_model()
  #single_worker_model.fit(x=train_datasets, epochs=3)

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

  ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32), (tf.TensorShape([28, 28, 1]), tf.TensorShape([1])))
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
