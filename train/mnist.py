"""MNIST CNN Network"""

from argparse import ArgumentParser
import os
import json
import tensorflow as tf
import model
import data
import json


def run(data_directory, model_directory):
    """Run training and evaluation
    Arguments:
        data_directory: directory where data is located
        model_directory: output directory where model and checkpoints will be placed
    """

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=20,
        save_summary_steps=20,
    )

    hparams = {
        'learning_rate': 1e-3,
        'dropout_rate': 0.4,
        'data_directory': data_directory
    }

    mnist_classifier = tf.estimator.Estimator(
        model_fn=model.head_model_fn,
        model_dir=model_directory,
        config=run_config,
        params=hparams
    )

    train_batch_size = 1024

    train_files = os.path.join(data_directory, 'train-*.tfrecords')
    train_input_fn = data.data_input_fn(
        train_files, batch_size=train_batch_size)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=400)

    eval_files = os.path.join(data_directory, 'validation.tfrecords')
    eval_input_fn = data.data_input_fn(eval_files, batch_size=1)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, start_delay_secs=60)

    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

    # Export for serving
    # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(data.get_feature_columns(with_label=False))
    # mnist_classifier.export_savedmodel(
    #     os.path.join(hparams['data_directory'], 'serving'),
    #     serving_input_receiver_fn
    # )


if __name__ == '__main__':

    PARSER = ArgumentParser()
    PARSER.add_argument(
        "--data-directory",
        help='Directory where TFRecords are stored'
    )
    PARSER.add_argument(
        '--model-directory',
        help='Directory where model summaries and checkpoints are stored'
    )
    ARGS = PARSER.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Environment variables')
    tf.logging.info(json.dumps(dict(os.environ), indent=4))

    run(**vars(ARGS))
