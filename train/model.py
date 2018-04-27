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
        pool2_flat = tf.contrib.layers.flatten(pool2)
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

    # Can just provide optimizer as of master head
    # return head.create_estimator_spec(
    #     features, mode, logits, labels, optimizer=tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    # )

    def train_op_fn(loss): return tf.train.AdamOptimizer(
        learning_rate=params['learning_rate']).minimize(loss, global_step=tf.train.get_global_step())
    return head.create_estimator_spec(
        features, mode, logits, labels, train_op_fn=train_op_fn
    )


def mnist_model(features, mode, params):
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
            features, [-1, 28, 28, 1], name='input_reshape')
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
        pool2_flat = tf.contrib.layers.flatten(pool2)
        dense = tf.layers.dense(
            inputs=pool2_flat, units=1024, activation=tf.nn.relu, trainable=is_training)
        dropout = tf.layers.dropout(
            inputs=dense, rate=params['dropout_rate'], training=is_training)

    with tf.name_scope('Predictions'):
        # Logits Layer
        logits = tf.layers.dense(
            inputs=dropout, units=10, trainable=is_training)

        return logits


def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN.
    Arguments:
        features: tensor of MNIST images
        labels: MNIST labels
        mode: estimator mode
        params: dictionary of hyperparameters

    Returns:
        EstimatorSpec
    """

    logits = mnist_model(features, mode, params)
    predicted_class = tf.argmax(input=logits, axis=1, output_type=tf.int32)
    scores = tf.nn.softmax(logits, name='softmax_tensor')

    # Generate Predictions
    predictions = {
        tf.contrib.learn.PredictionKey.CLASSES: predicted_class,
        tf.contrib.learn.PredictionKey.PROBABILITIES: scores
    }

    # PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        _classifier_output = tf.estimator.export.ClassificationOutput(
            scores=scores,
            classes=tf.cast(predicted_class, tf.string))

        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: _classifier_output,
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: _classifier_output,
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: scores,
            tf.saved_model.signature_constants.PREDICT_OUTPUTS: tf.estimator.export.PredictOutput(
                predictions)
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # TRAIN and EVAL
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predicted_class)
    eval_metric = {'accuracy': accuracy}

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('accuracy', accuracy[1])

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric,
        predictions=predictions)
