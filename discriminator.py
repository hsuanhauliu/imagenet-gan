import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub


def discriminator(x, label, model):
	disc = tf_hub.Module(model)
	logits = disc(x)
	probs = tf.nn.softmax(logits)
	
	select_class = np.zeros([1001, 1])
	select_class[label] = 1
	select_vector = tf.convert_to_tensor(select_class, dtype=np.float32)
	return tf.matmul(probs, select_vector)
	
