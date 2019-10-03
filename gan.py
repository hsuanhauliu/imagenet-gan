import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

import generator


def main():

	# import image
	label = 271
	img = cv.imread("imgs/wolf1.jpg")
	img = cv.resize(img, dsize=(224, 224))
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	img = img / 255
	"""
	plt.imshow(img)
	plt.show()
	"""
	img.shape = (1, 224, 224, 3)

	# Use discriminator
	discriminator = tf_hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3")
	print(f"Expecting input image size: {tf_hub.get_expected_image_size(discriminator)}")
	print(f"Expecting input image channels: {tf_hub.get_num_image_channels(discriminator)}")
	
	x = tf.placeholder(dtype=float, shape=[None, 224, 224, 3], name="discriminator_input")
	logits = discriminator(x)
	y = tf.nn.softmax(logits)

	# Build generator
	random_x = np.random.uniform(-1, 1, size=[1, 128])
	gen_x = tf.placeholder(dtype=float, shape=[None, 128], name="generator_input")
	gen_y = generator.dc_generator(gen_x)

	# Define training parameters
	#gen_loss = tf.reduce_mean(tf.
	#gen_train_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(gen_loss)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		fake_img = sess.run(gen_y, feed_dict={gen_x: random_x})
		
		plt.imshow(fake_img[0])
		plt.show()

		"""
		output = sess.run(y, feed_dict={x: img})
		output.shape = (-1)

		true_probability = output[label]
		false_probability = 1 - true_probability
		print(f'total: {np.sum(output)}')
		print(true_probability, false_probability)
		"""
		
		"""
		# scale the probabilities to be binary
		true_probability = true_probability * 1001
		false_probability = false_probability * 1001 / 1000
		sum_probability = true_probability + false_probability

		true_probability /= sum_probability
		false_probability /= sum_probability
		print(true_probability, false_probability)
		"""

	return


if __name__ == "__main__":
	main()
