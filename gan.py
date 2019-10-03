import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

from generator import dc_generator
from discriminator import discriminator


def train():
	""" Train generator """
	label = 271 # white wolf
	disc_model = "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3"
	batch_size = 64
	train_epochs = 2000

	# Specify discriminator
	disc_x = tf.placeholder(dtype=float, shape=[None, 224, 224, 3], name="discriminator_input")
	disc_y = discriminator(disc_x, label, disc_model)
	
	# Build generator
	gen_x = tf.placeholder(dtype=float, shape=[None, 128], name="generator_input")
	gen_y = dc_generator(gen_x)

	# Define training parameters
	gen_loss = tf.reduce_mean(disc_y)
	gen_train_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(gen_loss)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(train_epochs):
			noise_input = np.random.uniform(-1, 1, size=[batch_size, 128])
			sess.run(gen_train_opt, feed_dict={gen_x: noise_input})
			
			if not i % 10:
				print(gen_loss.eval({gen_x: noise_input}))


def test():
	""" Test Generator """
	gen_x = tf.placeholder(dtype=float, shape=[None, 128], name="generator_input")
	gen_y = dc_generator(gen_x)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		noise_input = np.random.uniform(-1, 1, size=[batch_size, 128])
		fake_img = sess.run(gen_y, feed_dict={gen_x: noise_input})
		plt.imshow(fake_img)
		plt.show()
	

if __name__ == "__main__":
	train()
