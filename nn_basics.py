""" Reusable network components """

import tensorflow as tf

class ff_layer: 
	def __init__(self, shape, layer_name, activation_func=tf.nn.relu, 
			   init_func=tf.random_normal_initializer(0, 3)):
		self.shape = shape
		self.layer_name = layer_name
		self.activation_func = activation_func
		self.init_func = init_func
		with tf.variable_scope(layer_name, initializer=init_func) as scope:
			with tf.name_scope('weights'):
				self.weights = tf.get_variable('weights', shape)
				tf.summary.scalar('weight', tf.reduce_mean(self.weights))
			with tf.name_scope('biases'):
				self.biases = tf.get_variable('biases', [1, shape[1]])
				tf.summary.scalar('bias', tf.reduce_mean(self.biases))
					
def ff_network(architecture, input, name, reuse_vars=False, activation_func=tf.nn.relu):
	with tf.variable_scope(name) as scope:
		if (reuse_vars):
			scope.reuse_variables()
		layers = []
		outputs = [input]
		for i in range(len(architecture) - 1):
			layers.append(ff_layer(architecture[i : i + 2], "".join([name, "_", str(i+1)]), activation_func))
			outputs.append(feedforward(layers[i], outputs[i]))
		return layers, outputs
					
def feedforward(ff_layer, input_tensor):
	name = "".join([ff_layer.layer_name, "_output"])
	with tf.name_scope(name):
		with tf.name_scope('weighted_input'):
			weighted_input = tf.matmul(input_tensor, ff_layer.weights) + ff_layer.biases
		with tf.name_scope('activation'):
			activation = ff_layer.activation_func(weighted_input)
			tf.summary.scalar('activation', tf.reduce_mean(activation))
		return activation