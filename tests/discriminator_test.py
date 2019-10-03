import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from discriminator import discriminator


def main():
	label = 271	# white wolf
	img = import_img("imgs/wolf1.jpg")

	disc_x = tf.placeholder(dtype=float, shape=[None, 224, 224, 3], name="discriminator_input")
	disc_y = discriminator(disc_x, label, "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3")
	

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		disc_output = sess.run(disc_y, feed_dict={disc_x: new_img})
		print(disc_output.shape)
		print(disc_output)


def import_img(path):
	img = cv.imread(path)
	img = cv.resize(img, dsize=(224, 224))
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	img = img / 255
	img.shape = (1, 224, 224, 3)
	return img


if __name__ == "__main__":
	main()
