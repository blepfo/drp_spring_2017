""" Reusable network components """

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from nn_basics import ff_network
from nn_basics import ff_layer
from nn_basics import feedforward

class VAE:
	def __init__(self, encode_architecture, dec_architecture, name="VAE"):
		if ((encode_architecture[-1] != dec_architecture[0]) or (encode_architecture[0] != dec_architecture[-1])):
			raise Exception("Bad architecture")
		self.encode_architecture = encode_architecture
		self.graph = tf.Graph()
		self.name = name
		self.hidden_dim = encode_architecture[-1]
		# Save model to ./saves/<name>
		self.save_dir = "".join(["./saves/", name])
		# Log model to ./logs/<name>
		self.log_dir = "".join(["./logs/", name])
		with self.graph.as_default():
			with tf.variable_scope(name):
				with tf.name_scope("inputs"):
					with tf.name_scope("images"):
						self.inputs = tf.placeholder(tf.float32, [None, encode_architecture[0]])
					with tf.name_scope("latent_samples"):
						self.input_samples = tf.placeholder(tf.float32, [None, self.hidden_dim])
				# ENCODER 
				with tf.name_scope("E"):
					encode_layers, encode_outputs = ff_network(encode_architecture[0:-1], self.inputs, "E_hidden")
					with tf.name_scope("encode_means"):
						encode_means_layer = ff_layer([encode_architecture[-2], self.hidden_dim], "encode_means")
						encode_means = feedforward(encode_means_layer, encode_outputs[-1])
					with tf.name_scope("encode_logvar"):
						encode_logvar_layer = ff_layer([encode_architecture[-2], self.hidden_dim], "encode_logvar")
						encode_logvar = feedforward(encode_logvar_layer, encode_outputs[-1])
				# LATENT
				with tf.name_scope("latent_training"):
					latent_training = tf.multiply(self.input_samples, tf.sqrt(tf.exp(encode_logvar))) + encode_means
				# DECODER
				with tf.name_scope("D_train"):
					decode_train_layers, decode_train_outputs = ff_network(dec_architecture, latent_training, "D")
				with tf.name_scope("D_gen"):
					dec_gen_layers, decode_gen_outputs = ff_network(dec_architecture, self.input_samples, "D", reuse_vars=True)
				self.generated = decode_gen_outputs[-1]
				with tf.name_scope("cost"):
					with tf.name_scope("mse"):
						mse = 0.5 * tf.reduce_mean(
							tf.reduce_sum(
								tf.square(decode_train_outputs[-1] - self.inputs), 1)
						)
						tf.summary.scalar('mse', mse)
					with tf.name_scope("kl_divergence"):
						kl_divergence = 0.5 * tf.reduce_mean(
							(tf.reduce_sum(tf.exp(encode_logvar), 1) +
							tf.reduce_sum(tf.square(encode_means), 1) - 
							self.hidden_dim -
							tf.reduce_sum(encode_logvar, 1)))
						tf.summary.scalar('kl_divergence', kl_divergence)
					cost = mse + kl_divergence
					tf.summary.scalar('cost', cost)
				with tf.name_scope("train"):
					self.train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cost)
			self.summaries = tf.summary.merge_all()
		# Initialize and save
		with tf.Session(graph=self.graph) as sess:
			sess.run(tf.global_variables_initializer())
			tf.train.Saver().save(sess, self.save_dir)

	def train(self, inputs, epoch_num):
		with tf.Session(graph=self.graph) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, self.save_dir)
			file_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
			latent_samples = np.random.randn(inputs.shape[0], self.encode_architecture[-1])
			_, summaries = sess.run([self.train_step, self.summaries], 
						feed_dict={self.inputs : inputs, self.input_samples : latent_samples})
			file_writer.add_summary(summaries, epoch_num)	
			saver.save(sess, self.save_dir)
					
	def generate(self):
		with tf.Session(graph=self.graph) as sess:
			tf.train.Saver().restore(sess, self.save_dir)
			latent_sample = np.random.randn(1, self.encode_architecture[-1])
			image = sess.run(self.generated, feed_dict={self.input_samples : latent_sample})
			plt.imshow(np.reshape(image, [28,28]), cmap="Greys")
			plt.axis('off')
			plt.show()
	
	def __str__(self):
		with self.graph.as_default():
			return "\n".join([var.name for var in tf.trainable_variables()])