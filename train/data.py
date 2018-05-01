"""MNIST Data"""

import tensorflow as tf
import os


def get_feature_columns(with_label=True):
    features = {
        'image_raw': tf.FixedLenFeature([], tf.string)
    }
    if with_label:
        features['label'] = tf.FixedLenFeature([], tf.int64)
    return features


def data_input_fn2(filenames, batch_size=1024, shuffle=False, buffer_size=4096, num_epochs=None):

    tf.contrib.data.AUTOTUNE = 1

    def _parser(record):
        image = tf.decode_raw(record['image_raw'], tf.float32)

        label = tf.cast(record['label'], tf.int32)

        return {
            'image': image
        }, label

    def _input_fn():
        dataset = tf.contrib.data.make_batched_features_dataset(
            filenames, batch_size, get_feature_columns(),
            shuffle=shuffle, shuffle_buffer_size=buffer_size,
            num_epochs=num_epochs, sloppy_ordering=True
        )

        dataset = dataset.map(_parser, num_parallel_calls=os.cpu_count())
        return dataset

    return _input_fn
