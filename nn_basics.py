"""Reusable feedforward network components """

import tensorflow as tf

class ff_layer:
	"""Single layer in a feedforward network. 
	
	Attributes:
		weights (tensor): Weights used to compute activation
		biases (tensor): Biases used to compute activation
		shape (int tuple): Tuple of (input_shape, output_shape)
		layer_name (string): Name of the network 
		activation_func (tf actvation function): TensorFlow activation function
		init_func (tf initializer function): Tensorflow initalizer function
		
	"""
	
	def __init__(self, shape, layer_name, activation_func=tf.nn.relu,
				init_func=tf.random_uniform_initializer(-1, 1)):
		""" Constructor.
		
		Args:
			shape (int tuple): Tuple of (input_shape, output_shape)
			layer_name (string): Name of the network 
			activation_func (tf actvation function): TensorFlow activation function
			init_func (tf initializer function): Tensorflow initalizer function
			
		Returns:
			ff_layer object.
		"""
		self.shape = shape 
		self.layer_name = layer_name
		self.activation_func = activation_func
		with tf.variable_scope(layer_name, initializer=init_func):
			with tf.name_scope('weights'):
				self.weights = tf.get_variable('weights', shape)
				tf.summary.scalar('weight', tf.reduce_mean(self.weights))
				tf.summary.histogram('weights', self.weights)
			with tf.name_scope('biases'):
				self.biases = tf.get_variable('biases', [1, shape[1]])
				tf.summary.scalar('biases', tf.reduce_mean(self.biases))
				
	def feedforward(self, input):
		""" Compute network output.
		
		Args:
			input (tensor): Tensor of shape (batch_num, x) to feed through the layer
		
		Returns:
			Tensor of layer activations when the given input is fed through the layer
		"""
		with tf.name_scope(self.layer_name):
			activation = self.activation_func(tf.matmul(input,self.weights) + self.biases)
			tf.summary.scalar('activation', tf.reduce_mean(activation))
			return activation
			
class ff_network:
	""" Feedforward network consisting of several feedforward layers.
	
	Attributes:
		architecture (int tuple): Tuple or list where entry i is number of neurons in layer id
		name (string): Name for the network
		activation_funcs (list of tf activation functions): Activation functions for each layer
		layers (ff_layer list): List of ff_layers in the network
	"""
	
	def __init__(self, architecture, name, activation_funcs=None):
		""" Constructor.
		
		Args:
			architecture (int tuple): Tuple or list where entry i is number of neurons in layer id
			name (string): Name for the network
			activation_funcs (list of tf activation functions): Activation functions for each layer
		
		Returns: 
			ff_network object.
		"""
		self.architecture = architecture
		self.name = name
		if (activation_funcs == None):
			activation_funcs = [tf.sigmoid] * len(architecture)
		with tf.name_scope(name):
			layers = [ff_layer(architecture[i : i + 2], "%s_%d" % (name, (i+1)), activation_funcs[i]) for i in range(len(architecture) - 1)]
		self.layers = layers
	
	def compute_output(self, input):
		""" Feed an input through the network to produce output.
		
		Args: 
			input (tensor): input to feed through the network of shape (batch_num, x)
			
		Returns:
			List of output activation tensors for each layer 
		"""
		with tf.name_scope(self.name):
			outputs = [input]
			for i, layer in enumerate(self.layers):
				outputs.append(layer.feedforward(outputs[i]))
			return outputs