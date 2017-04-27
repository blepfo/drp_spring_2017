""" Reusable feedforward network components """

import tensorflow as tf

class ff_layer:
	def __init__(self, shape, layer_name, activation_func=tf.nn.relu,
				init_func=tf.random_uniform_initializer(-1, 1)):
		self.shape = shape 
		self.layer_name = layer_name
		self.activation_func = activation_func
		with tf.variable_scope(layer_name, initializer=init_func):
			with tf.name_scope('weights'):
				self.weights = tf.get_variable('weights', shape)
				# tf.Variable(tf.zeros(shape), name='weights', dtype=tf.float32)
				tf.summary.scalar('weight', tf.reduce_mean(self.weights))
				tf.summary.histogram('weights', self.weights)
			with tf.name_scope('biases'):
				self.biases = tf.get_variable('biases', [1, shape[1]])
				# self.biases = tf.Variable(tf.zeros([1, shape[1]]), name='biases', dtype=tf.float32)
				tf.summary.scalar('biases', tf.reduce_mean(self.biases))
				
	def feedforward(self, input):
		with tf.name_scope(self.layer_name):
			activation = self.activation_func(tf.matmul(input,self.weights) + self.biases)
			tf.summary.scalar('activation', tf.reduce_mean(activation))
			return activation
			
class ff_network:
	def __init__(self, architecture, name, activation_funcs=None):
		self.name = name
		self.architecture = architecture
		if (activation_funcs == None):
			activation_funcs = [tf.sigmoid] * len(architecture)
		with tf.name_scope(name):
			layers = [ff_layer(architecture[i : i + 2], "%s_%d" % (name, (i+1)), activation_funcs[i]) for i in range(len(architecture) - 1)]
		self.layers = layers
	
	def compute_output(self, input):
		with tf.name_scope(self.name):
			outputs = [input]
			for i, layer in enumerate(self.layers):
				outputs.append(layer.feedforward(outputs[i]))
			return outputs

