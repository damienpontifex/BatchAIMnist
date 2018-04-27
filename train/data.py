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
    
    dataset = tf.contrib.data.make_batched_features_dataset(
        filenames, batch_size, get_feature_columns(), 
        shuffle=shuffle, shuffle_buffer_size=buffer_size, 
        num_epochs=num_epochs
    )

    def _parser(record):
        image = tf.decode_raw(record['image_raw'], tf.float32)

        label = tf.cast(record['label'], tf.int32)

        return {
            'image': image
        }, label

    dataset = dataset.map(_parser, num_parallel_calls=os.cpu_count())
    iterator = dataset.make_one_shot_iterator()
    return lambda: iterator.get_next()
    # return lambda: dataset

def data_input_fn(filenames, batch_size=1024, shuffle=False, buffer_size=4096, num_epochs=None):
    """Construct an estimator input_fn
    Arguments:
        filenames: tfrecord files to be used
        batch_size: batch size
        shuffle: whether to shuffle records
    Returns:
        Callable function confirming to tf estimator input_fn
    """
    def _parser(record):
        
        parsed_record = tf.parse_single_example(record, get_feature_columns())
        image = tf.decode_raw(parsed_record['image_raw'], tf.float32)

        label = tf.cast(parsed_record['label'], tf.int32)

        # return image, tf.one_hot(label, depth=10)
        return {
            'image': image
        }, label
        
    def _input_fn():
        files = tf.data.Dataset.list_files(filenames)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=32)
        
        if shuffle:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs)
            )
        else:
            dataset = dataset.repeat(num_epochs)# Infinite iterations: let experiment determine num_epochs

        dataset = dataset.map(_parser, num_parallel_calls=os.cpu_count())
        dataset = dataset.batch(batch_size).prefetch(buffer_size)

        return dataset

    return _input_fn