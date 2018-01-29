# Well, this is based on the original work by the TensorFlow team:
# tensorflow/examples/tutorials/mnist/mnist_deep.py
# I believe they should get their due credit for that work.
# This is new work.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import csv

from time import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import os
import tensorflow as tf
# suppress most of tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import itertools as it

FLAGS = None


def deepnn(x, cnn, nn):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
      cnn: CNN pipeline architecture: filter size, output depth, scale factor
      nn: NN pipeline architecture - fully connected: output depth, relu

    Returns: y, a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9).
    """

    size = int(28)

    # Reshape to use within a convolutional neural net.
    with tf.name_scope('reshape'):
        z = tf.reshape(x, [-1, size, size, 1])

    z, size, odepth = CNN_pipeline(z, size, cnn)

    size = size * size * odepth
    z = tf.reshape(z, [-1, size])

    z = NN_pipeline(z, size, nn)

    return z


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def layerConv(name, x, fsize, idepth, odepth):
    with tf.name_scope(name):
        w = weight_variable([fsize, fsize, idepth, odepth])
        b = bias_variable([odepth])
        z = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b

        z = tf.nn.relu(z)

    return z


def layerResize(name, x, isize, factor):
    if factor == 1:
        return x, isize
    with tf.name_scope(name):
        osize = int(isize / factor)
        z = tf.image.resize_bilinear(x, [osize, osize], True)
    return z, osize


def CNN_pipeline(x, isize, cnn):
    z = x
    size = isize
    idepth = 1
    odepth = 1
    for l in range(len(cnn)):
        idepth = odepth
        fsize = cnn[l][0]
        odepth = cnn[l][1]
        factor = cnn[l][2]

        z = layerConv('conv%d' % l, z, fsize, idepth, odepth)
        z, size = layerResize('resize%d' % l, z, size, factor)

    return z, size, odepth


def layerNN(name, x, isize, osize, relu):
    with tf.name_scope(name):
        w = weight_variable([isize, osize])
        b = bias_variable([osize])
        z = tf.matmul(x, w) + b

        if relu == 1:
            z = tf.nn.relu(z)

    return z


def NN_pipeline(x, size, nn):
    z = x
    osize = size
    for l in range(len(nn)):
        isize = osize
        osize = nn[l][0]
        relu = nn[l][1]

        z = layerNN('nn%d' % l, z, isize, osize, relu)

    return z


def main(mnist, cnn, nn, steps):
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv = deepnn(x, cnn, nn)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    testAccuracy = 0
    start_time = time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            batch = mnist.train.next_batch(50)
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x: batch[0], y_: batch[1]})
                # print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(
                feed_dict={x: batch[0], y_: batch[1]})

        testAccuracy = accuracy.eval(
            feed_dict={x: mnist.test.images, y_: mnist.test.labels})

    end_time = time()
    time_taken = end_time - start_time  # time_taken is in seconds

    print('%7d %11.3f %5d %s %s' % (time_taken, testAccuracy, steps, cnn, nn))
    tf.reset_default_graph()


# CNN pipeline architecture
def GenCNNs(filterSizes, oDepths, scaleFactors, nrCNNLayers):
    cnnLayers = [list(cnnLayer) for cnnLayer in it.product(
        filterSizes, oDepths, scaleFactors)]
    cnns = [list(cnn) for cnn in it.product(cnnLayers, repeat=nrCNNLayers)]
    return cnns


# NN pipeline architecture - fully connected
def GenNNs(nnLayerSizes, activation, nrNNLayers):
    nnLayers = [list(item) for item in it.product(nnLayerSizes, Activation)]
    nns = [list(item) for item in it.product(nnLayers, repeat=nrNNLayers)]

    # add final softmax layer
    for nn in nns:
        nn.append([10, 0])
    return nns


if __name__ == '__main__':
    steps = 50

    # generate all CNN architectures with 1 and 2 layers
    FilterSizes = [3]
    ODepths = [32]
    ScaleFactors = [1]
    NrCNNLayers = [1, 2]

    # generate all NN architectures with 0 and 1 hidden layers
    NNLayerSizes = [128]
    Activation = [1]  # Relu only for now
    NrNNLayers = [0, 1]

    CNNs = []
    for n in NrCNNLayers:
        CNNs = CNNs + GenCNNs(FilterSizes, ODepths, ScaleFactors, n)
    NNs = []
    for n in NrNNLayers:
        NNs = NNs + GenNNs(NNLayerSizes, Activation, n)

    # print("CNNs:")
    # for cnn in CNNs:
    #     print(cnn)

    # print("NNs:")
    # for nn in NNs:
    #     print(nn)

    architectures = [list(item) for item in it.product(CNNs, NNs)]
    print("Nr. networks: %3d" % (len(architectures)))
    print('')
    architectures = sorted(architectures, key=len)

    # Import data
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    print("Time(s) Accuracy(%) Steps Architecture")
    for arch in architectures:
        main(mnist, arch[0], arch[1], steps)
    print('')

