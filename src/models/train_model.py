#!/usr/bin/env python

""" An implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import vgg_model
import utils

# parameters to manage experiments
STYLE_IMAGE = 'data/styles/{}.jpg'
CONTENT_IMAGE = 'data/content/{}.jpg'
IMAGE_HEIGHT = 333
IMAGE_WIDTH = 333
IMAGE_FILENAME = '{}/{}.png'

tf.flags.DEFINE_string(
    "style", 'hokusai', "Style image. Image must be put in data/styles.")
tf.flags.DEFINE_string(
    "content", None, 'Content image. Image must be put in data/content.')
tf.flags.DEFINE_float(
    "noise_ratio", 0.6, "Percentage of weight of the noise for intermixing with the content image")
tf.flags.DEFINE_float(
    "style_loss_ratio", 0.05, "The weight of style loss with content loss. The total loss is content_loss + ratio * style_loss")
tf.flags.DEFINE_string(
    "content_loss_layer", "conv4_2", "The layer used for calculating content loss. (conv1_2, conv2_2, conv3_2, conv4_2, conv5_2)")
tf.flags.DEFINE_string(
    "output_dir", "data/outputs", "The output dir where generated image will be put.")
tf.flags.DEFINE_integer(
    "iters", 300, "Iterations")
tf.flags.DEFINE_float(
    "learning_rate", 2.0, "Learning rate")

FLAGS = tf.flags.FLAGS

# Layers used for style features. You can change this.
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
W = [0.5, 1.0, 1.5, 3.0, 4.0] # give more weights to deeper layers.

MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
""" MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering.
The input images should be zero-centered by mean pixel (rather than mean image) subtraction.
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
"""

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'models/imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783

def _create_content_loss(p, f):
    """
    Calculate the loss between the feature representation of the
    content image and the generated image.

    Inputs:
        p, f are just P, F in the paper
        (read the assignment handout if you're confused)
        Note: we won't use the coefficient 0.5 as defined in the paper
        but the coefficient as defined in the assignment handout.
    Output:
        the content loss

    """
    return tf.reduce_mean(tf.square(f - p)) / (4 * p.size)

def _gram_matrix(F, N, M):
    """
    Create and return the gram matrix for tensor F
    """
    m = tf.reshape(F, shape=[M, N])
    return tf.matmul(tf.transpose(m), m)

def _single_style_loss(a, g):
    """ Calculate the style loss at a certain layer
    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image
    Output:
        the style loss at a certain layer (which is E_l in the paper)

    """
    N = a.shape[3]
    M = a.shape[1] * a.shape[2]
    A = _gram_matrix(a, N, M)
    G = _gram_matrix(g, N, M)

    return tf.reduce_mean(tf.square(G - A)) / (4 * N * N * M * M)


def _create_style_loss(A, model):
    """
    Return the total style loss
    """
    n_layers = len(STYLE_LAYERS)
    E = [W[i] * _single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]
    return tf.reduce_sum(E)

def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            p = sess.run(model[FLAGS.content_loss_layer])
        content_loss = _create_content_loss(p, model[FLAGS.content_loss_layer])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])
        style_loss = _create_style_loss(A, model)

        total_loss = content_loss + FLAGS.style_loss_ratio * style_loss

    return content_loss, style_loss, total_loss

def _create_summary(model):
    """ Create summary ops necessary
        Hint: don't forget to merge them
    """
    tf.summary.scalar('content_loss', model['content_loss'])
    tf.summary.scalar('style_loss', model['style_loss'])
    tf.summary.scalar('total_loss', model['total_loss'])
    return tf.summary.merge_all()

def train(model, generated_image, initial_image):
    """
    Train your model.
    """
    skip_step = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter('graphs/style_transfer', sess.graph)
        sess.run(generated_image.assign(initial_image))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('graphs/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()

        start_time = time.time()
        for index in range(initial_step, FLAGS.iters):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20

            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:
                total_loss, gen_image = sess.run([model['total_loss'], generated_image])
                gen_image = gen_image + MEAN_PIXELS
                summary = sess.run(model['summary_op'])
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = IMAGE_FILENAME.format(FLAGS.output_dir,  index)
                utils.save_image(filename, gen_image)

                if (index + 1) % 20 == 0:
                    saver.save(sess, 'graphs/checkpoints/style_transfer', index)

def main(argv):
    with tf.variable_scope('input') as scope:
        # use variable instead of placeholder because we're training the intial image to make it
        # look like both the content image and the style image
        input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)

    utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
    model = vgg_model.load_vgg(VGG_MODEL, input_image)
    model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    content_image = utils.get_resized_image(CONTENT_IMAGE.format(FLAGS.content), IMAGE_HEIGHT, IMAGE_WIDTH)
    content_image = content_image - MEAN_PIXELS
    style_image = utils.get_resized_image(STYLE_IMAGE.format(FLAGS.style), IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image = style_image - MEAN_PIXELS

    model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, input_image, content_image, style_image)
    model['optimizer'] = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(model['total_loss'])
    model['summary_op'] = _create_summary(model)

    initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, FLAGS.noise_ratio)
    train(model, input_image, initial_image)

if __name__ == '__main__':
    tf.app.run()
