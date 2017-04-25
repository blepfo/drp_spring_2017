""" Reusable network components """

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from math import floor


def feedforward(input, shape, layer_name, activation_func=tf.nn.relu,  
			   init_func=tf.random_uniform_initializer(-1, 1), reuse_vars=False):
	with tf.variable_scope(layer_name, initializer=init_func) as scope:
		if (reuse_vars):
			scope.reuse_variables()
		weights = tf.get_variable('weights', shape)
		tf.summary.scalar('weights', tf.reduce_mean(weights))
		biases = tf.get_variable('biases', [1, shape[1]])
		tf.summary.scalar('biases', tf.reduce_mean(biases))
		activation = activation_func(tf.matmul(input, weights) + biases)
		tf.summary.scalar('activation', tf.reduce_mean(activation))
		return activation
		
def ff_network(input, architecture, name, activation_funcs=None, reuse_vars=False):
	if (activation_funcs == None):
		activation_funcs = [tf.sigmoid] * len(architecture)
	with tf.variable_scope(name) as scope:
		if (reuse_vars):
			scope.reuse_variables()
		activations = [input]
		for i in range(len(architecture) - 1):
			activations.append(feedforward(activations[i], architecture[i : i + 2], "%s_%d" % (name, (i+1)), activation_funcs[i], reuse_vars=reuse_vars))
		return activations
		
def generate(model):
	""" Accepts GAN or VAE object and generates three sample images
	Assumes we can access model.graph, model.gen_architecture, 
	model has op model.generated,
	and that model is saved in ./<model.save_dir> """
	with tf.Session(graph=model.graph) as sess:
		tf.train.Saver().restore(sess, model.save_dir)
		samples = np.random.randn(3, model.hidden_dim)
		images = sess.run(model.generated, feed_dict={model.latent_samples : samples})
		for i in range(3):
			plt.subplot(1, 3, i + 1)
			plt.imshow(np.reshape(images[i], [28,28]), cmap="Greys")
			plt.axis('off')
	plt.show()

