"""
	DC Generator architecture inspired by:
		https://towardsdatascience.com/image-generator-drawing-cartoons-with-generative-adversarial-networks-45e814ca9b6b
"""

import numpy as np
import tensorflow as tf


def dc_generator(x, training=False):
	""" Deep Convo GAN """

	# 14x14x1024
	fc_1 = tf.layers.dense(x, 14 * 14 * 1024)
	fc_1 = tf.reshape(fc_1, (-1, 14, 14, 1024))
	fc_1 = tf.nn.leaky_relu(fc_1)

	# 14x14x1024 -> 28x28x512
	trans_conv_1 = _conv2d_trans(fc_1, 512, "trans_conv_1", training=training)

	# 28x28x512 -> 56x56x256
	trans_conv_2 = _conv2d_trans(trans_conv_1, 256, "trans_conv_2", training=training)

	# 56x56x256 -> 112x112x128
	trans_conv_3 = _conv2d_trans(trans_conv_2, 128, "trans_conv_3", training=training)

	# 112x128 -> 224x224x64
	trans_conv_4 = _conv2d_trans(trans_conv_3, 64, "trans_conv_4", training=training)

	# 128x128x64 -> 128x128x3
	logits = tf.layers.conv2d_transpose(inputs=trans_conv_4,
										filters=3,
										kernel_size=[5, 5],
										strides=[1, 1],
										padding="SAME",
										kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
										name="logits")
	y = tf.tanh(logits, name="out")
	return y


def _conv2d_trans(x, filter_size, layer_name, training):
	trans_conv = tf.layers.conv2d_transpose(inputs=x,
											filters=filter_size,
											kernel_size=[5,5],
											strides=[2,2],
											padding="SAME",
											kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
											name=layer_name)

	batch_trans_conv = tf.layers.batch_normalization(inputs=trans_conv,
													 training=training,
													 epsilon=0.00005)

	trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv)

	return trans_conv1_out
