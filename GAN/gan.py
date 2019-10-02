import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub


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

	discriminator = tf_hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3")
	print(f"Expecting input image size: {tf_hub.get_expected_image_size(discriminator)}")
	print(f"Expecting input image channels: {tf_hub.get_num_image_channels(discriminator)}")
	
	x = tf.placeholder(dtype=float, shape=[None, 224, 224, 3])
	logits = discriminator(x)
	y = tf.nn.softmax(logits)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		output = sess.run(y, feed_dict={x: img})
		output.shape = (-1)

		true_probability = output[label]
		false_probability = 1 - true_probability
		print(f'total: {np.sum(output)}')
		print(true_probability, false_probability)
		
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
