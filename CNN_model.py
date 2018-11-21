from tflearn.layers.conv import conv_2d, max_pool_2d, residual_block, batch_normalization
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn
import tensorflow as tf
import scipy


def cnn(img_size, lr):
    tf.reset_default_graph()

    convnet = input_data(shape=[None, img_size, img_size, 1], name='input')
    # conv layer 1 w/max pooling
    conv1 = conv_2d(convnet, 32, 2, activation='relu')
    conv1 = max_pool_2d(conv1, 2)
    # conv layer 2 w/max pooling etc
    conv2 = conv_2d(conv1, 64, 2, activation='relu')
    conv2 = max_pool_2d(conv2, 2)

    conv3 = conv_2d(conv2, 64, 2, activation='relu')
    conv3 = max_pool_2d(conv3, 2)

    conv4 = conv_2d(conv3, 128, 2, activation='relu')
    conv4 = max_pool_2d(conv4, 2)

    conv5 = conv_2d(conv4, 128, 2, activation='relu')
    conv5 = max_pool_2d(conv5, 2)

    conv6 = conv_2d(conv5, 256, 2, activation='relu')
    conv6 = max_pool_2d(conv6, 2)

    conv7 = conv_2d(conv6, 256, 2, activation='relu')
    conv7 = max_pool_2d(conv7, 2)

    conv8 = conv_2d(conv7, 512, 2, activation='relu')
    conv8 = max_pool_2d(conv8, 2)
    # fully connected layer
    fc1 = fully_connected(conv8, 1024, activation='relu')
    fc1 = dropout(fc1, 0.8)
    # fc2
    fc2 = fully_connected(fc1, 128, activation='relu')
    fc2 = dropout(fc2, 0.8)
    # output layer for classification
    output = fully_connected(fc2, 2, activation='softmax')
    output = regression(output, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(output, tensorboard_dir='log')     # logs to temp file for tensorboard analysis
    return model


def resnet(img_size, lr, n):
    tf.reset_default_graph()

    net = input_data(shape=[None, img_size, img_size, 1], name='input')

    conv1 = conv_2d(net, 16, 1, regularizer='L2', weight_decay=0.0001)
    res1 = residual_block(conv1, n, 16)
    res2 = residual_block(res1, 1, 32, downsample=True)
    res3 = residual_block(res2, n - 1, 32)
    res4 = residual_block(res3, 1, 64, downsample=True)
    res5 = residual_block(res4, n - 1, 64)
    batch_norm = batch_normalization(res5)
    activ = tflearn.activation(batch_norm, 'relu')
    gap = tflearn.global_avg_pool(activ)
    # Regression
    fc1 = tflearn.fully_connected(gap, 2, activation='softmax')
    mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    output = tflearn.regression(fc1, optimizer=mom, learning_rate=lr, loss='categorical_crossentropy')
    # Training
    model = tflearn.DNN(output, checkpoint_path='model_resnet32-basic',
                        max_checkpoints=2, tensorboard_verbose=0,
                        clip_gradients=0., tensorboard_dir='log')
    return model


def conv_res_integrated(img_size, lr, n, img_aug):
    tf.reset_default_graph()

    convnet = input_data(shape=[None, img_size, img_size, 1], name='input', data_augmentation=img_aug)
    # conv layer 1 w/max pooling
    conv1 = conv_2d(convnet, 32, 2, activation='relu', regularizer='L2', weight_decay=0.0001)
    conv1 = max_pool_2d(conv1, 2)
    # conv layer 2 w/max pooling etc
    conv2 = conv_2d(conv1, 32, 2, activation='relu', regularizer='L2', weight_decay=0.0001)
    conv2 = max_pool_2d(conv2, 2)

    conv3 = conv_2d(conv2, 64, 2, activation='relu', regularizer='L2', weight_decay=0.0001)
    conv3 = max_pool_2d(conv3, 2)

    conv4 = conv_2d(conv3, 64, 2, activation='relu', regularizer='L2', weight_decay=0.0001)
    conv4 = max_pool_2d(conv4, 2)
    # residual block
    res1 = residual_block(conv4, n, 128, downsample=True, regularizer='L2', weight_decay=0.0001)
    batch_norm = batch_normalization(res1)
    activ = tflearn.activation(batch_norm, 'relu')
    gap = tflearn.global_avg_pool(activ)
    # fully connected layer 1
    fc1 = fully_connected(gap, 1024, activation='relu', regularizer='L2', weight_decay=0.0001)
    fc1 = dropout(fc1, 0.85)
    # fully connected layer 2
    fc2 = tflearn.fully_connected(fc1, 2, activation='softmax')
    # output layer
    mom = tflearn.Momentum(0.1, lr_decay=0.01, decay_step=32000, staircase=True)
    output = tflearn.regression(fc2, optimizer=mom, learning_rate=lr, loss='categorical_crossentropy')
    # Training
    model = tflearn.DNN(output, checkpoint_path='model_integrated',
                        max_checkpoints=2, tensorboard_verbose=0,
                        tensorboard_dir='log', clip_gradients=0.)
    return model