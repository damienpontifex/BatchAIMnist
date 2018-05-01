#!/usr/bin/env python3
"""MNIST CNN Network"""

from argparse import ArgumentParser
import os
import json
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import model
import data
import json

tf.app.flags.DEFINE_string('data_directory', '',
                           'Directory where TFRecords are stored')
tf.app.flags.DEFINE_string('model_directory', '',
                           'Directory where model summaries and checkpoints are stored')
tf.app.flags.DEFINE_float('learning_rate', 0.4, '')
tf.app.flags.DEFINE_integer('batch_size', 1024, '')
tf.app.flags.DEFINE_integer('max_steps', 400, '')
tf.app.flags.DEFINE_bool('debug', False, '')

FLAGS = tf.app.flags.FLAGS


def main(_):
    """Run training and evaluation
    Arguments:
        data_directory: directory where data is located
        model_directory: output directory where model and checkpoints will be placed
    """

    run_config = tf.estimator.RunConfig()

    hparams = {
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': 0.4,
        'data_directory': FLAGS.data_directory
    }

    mnist_classifier = tf.estimator.Estimator(
        model_fn=model.head_model_fn,
        model_dir=FLAGS.model_directory,
        config=run_config,
        params=hparams
    )

    debug_hook = tf_debug.TensorBoardDebugHook("localhost:2333")

    hooks = [debug_hook] if FLAGS.debug else []

    train_files = os.path.join(FLAGS.data_directory, 'train-*.tfrecords')
    train_input_fn = data.data_input_fn(
        train_files, batch_size=FLAGS.batch_size)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=FLAGS.max_steps, hooks=hooks)

    eval_files = os.path.join(FLAGS.data_directory, 'validation.tfrecords')
    eval_input_fn = data.data_input_fn(eval_files, batch_size=1)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn)

    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

    # Export for serving
    # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(data.get_feature_columns(with_label=False))
    # mnist_classifier.export_savedmodel(
    #     os.path.join(hparams['data_directory'], 'serving'),
    #     serving_input_receiver_fn
    # )


if __name__ == '__main__':
    if 'TF_CONFIG' in os.environ:
        tf_config = os.environ['TF_CONFIG']
        tf_config_json = json.loads(tf_config)
        tf.logging.info('TF_CONFIG Environment value:')
        tf.logging.info(json.dumps(tf_config_json, indent=4))

    tf.app.run()
