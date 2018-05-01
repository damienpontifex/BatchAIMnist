"""MNIST Model"""

import tensorflow as tf


def head_model_fn(features, labels, mode, params):
    """CNN architecture to process 28x28x1 MNIST images
    Arguments:
        features: tensor of MNIST images
        mode: estimator mode
        params: dictionary of hyperparameters

    Returns:
        Tensor of the final layer output without activation
    """

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.name_scope('Input'):
        # Input Layer
        input_layer = tf.reshape(
            features['image'], [-1, 28, 28, 1], name='input_reshape')
        tf.summary.image('input', input_layer)

    with tf.name_scope('Conv_1'):
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=(2, 2), strides=2, padding='same')

    with tf.name_scope('Conv_2'):
        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training)

        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=(2, 2), strides=2, padding='same')

    with tf.name_scope('Dense_Dropout'):
        # Dense Layer
        # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        pool2_flat = tf.layers.flatten(pool2)
        dense = tf.layers.dense(
            inputs=pool2_flat, units=1024, activation=tf.nn.relu, trainable=is_training)
        dropout = tf.layers.dropout(
            inputs=dense, rate=params['dropout_rate'], training=is_training)

    with tf.name_scope('Predictions'):
        # Logits Layer
        logits = tf.layers.dense(
            inputs=dropout, units=10, trainable=is_training)

    # vocab = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    head = tf.contrib.estimator.multi_class_head(
        n_classes=10)  # , label_vocabulary=vocab)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    return head.create_estimator_spec(
        features, mode, logits, labels, optimizer=optimizer,
    )
