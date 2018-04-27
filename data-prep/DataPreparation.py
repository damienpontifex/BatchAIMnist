#! /usr/env/bin python3

"""Convert MNIST Dataset to local TFRecords"""

import argparse
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def _data_path(data_directory: str, name: str) -> str:
    """Construct a full path to a TFRecord file to be stored in the 
    data_directory. Will also ensure the data directory exists

    Args:
        data_directory: The directory where the records will be stored
        name:           The name of the TFRecord

    Returns:
        The full path to the TFRecord file
    """
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    return os.path.join(data_directory, '{}.tfrecords'.format(name))


def _int64_feature(value: int) -> tf.train.Features.FeatureEntry:
    """Create a Int64List Feature

    Args:
        value: The value to store in the feature

    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: str) -> tf.train.Features.FeatureEntry:
    """Create a BytesList Feature

    Args:
        value: The value to store in the feature

    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name: str, data_directory: str, num_shards: int=1):
    """Convert the dataset into TFRecords on disk

    Args:
        data_set:       The MNIST data set to convert
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
    """

    num_examples, rows, cols, depth = data_set.images.shape

    data_set = list(zip(data_set.images, data_set.labels))

    def _process_examples(example_dataset, filename: str):
        print('Processing {} data'.format(filename))
        dataset_length = len(example_dataset)
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index, (image, label) in enumerate(example_dataset):
                sys.stdout.write('\rProcessing sample {} of {}'.format(
                    index + 1, dataset_length))
                sys.stdout.flush()

                image_raw = image.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'label': _int64_feature(int(label)),
                    'image_raw': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())
            print()

    if num_shards == 1:
        _process_examples(data_set, _data_path(data_directory, name))
    else:
        sharded_dataset = np.array_split(data_set, num_shards)
        for shard, dataset in enumerate(sharded_dataset):
            _process_examples(dataset, _data_path(
                data_directory, '{}-{}'.format(name, shard + 1)))


def convert_to_tf_record(data_directory: str):
    """Convert the TF MNIST Dataset to TFRecord formats

    Args:
        data_directory: The directory where the TFRecord files should be stored
    """

    mnist = input_data.read_data_sets(
        "/tmp/tensorflow/mnist/input_data",
        reshape=False
    )

    convert_to(mnist.validation, 'validation', data_directory)
    convert_to(mnist.train, 'train', data_directory, num_shards=10)
    convert_to(mnist.test, 'test', data_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-directory',
        default='~/data/mnist',
        help='Directory where TFRecords will be stored')

    tf.logging.info('Environment variables')
    tf.logging.info(json.dumps(dict(os.environ), indent=4))

    args = parser.parse_args()
    convert_to_tf_record(os.path.expanduser(args.data_directory))
