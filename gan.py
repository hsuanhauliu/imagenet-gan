import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

from generator import dc_generator
from discriminator import discriminator


def train():
	label = 271 # white wolf
	disc_model = "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3"
	batch_size = 64

	# Specify discriminator
	disc_x = tf.placeholder(dtype=float, shape=[None, 224, 224, 3], name="discriminator_input")
	disc_y = discriminator(disc_x, label, disc_model)
	
	# Build generator
	gen_x = tf.placeholder(dtype=float, shape=[None, 128], name="generator_input")
	gen_y = dc_generator(gen_x)

	# Define training parameters
	#gen_loss = tf.reduce_mean(tf.
	#gen_train_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(gen_loss)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		noise_input = np.random.uniform(-1, 1, size=[batch_size, 128])
		fake_img = sess.run(gen_y, feed_dict={gen_x: noise_input})
		print(fake_img.shape)
		
	return


def test():
	pass


if __name__ == "__main__":
	train()
