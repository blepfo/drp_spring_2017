""" Reusable network components """

import tensorflow as tf
import numpy as np
from gen_model import Gen_Model
from nn_basics import ff_network
from nn_basics import ff_layer

class VAE(Gen_Model):
	def __init__(self, encode_architecture, gen_architecture, name="VAE"):
		if ((encode_architecture[-1] != gen_architecture[0]) or (encode_architecture[0] != gen_architecture[-1])):
			raise Exception("Bad architecture")
		super().__init__(name)
		self.encode_architecture = encode_architecture
		self.gen_architecture = gen_architecture
		self.graph = tf.Graph()
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
						self.latent_samples = tf.placeholder(tf.float32, [None, self.hidden_dim])
				# ENCODER 
				with tf.name_scope("E"):
					if (len(encode_architecture) == 2):
						e_hidden_out = self.inputs
					else:
						encode_hidden = ff_network(encode_architecture[0:-1], "E_H")
						e_hidden_out = encode_hidden.compute_output(self.inputs)[-1]
					# encode_means
					encode_means_layer = ff_layer([encode_architecture[-2], self.hidden_dim], 'encode_means', tf.identity)
					encode_means = encode_means_layer.feedforward(e_hidden_out)
					# encode_logvar
					encode_logvar_layer = ff_layer([encode_architecture[-2], self.hidden_dim], 'encode_logvar', tf.identity)
					encode_logvar = encode_logvar_layer.feedforward(e_hidden_out)
					
				# LATENT
				with tf.name_scope("latent_training"):
					latent_training = tf.multiply(self.latent_samples, tf.sqrt(tf.exp(encode_logvar))) + encode_means
				# DECODER
				dec_activations = [tf.sigmoid] * (len(gen_architecture) - 1)
				dec_activations[-1] = tf.identity
				dec_network = ff_network(gen_architecture, "D", activation_funcs=dec_activations)
				with tf.name_scope("D_train"):
					dec_train_out = dec_network.compute_output(latent_training)
				with tf.name_scope("D_gen"):
					dec_gen_out = dec_network.compute_output(self.latent_samples)
				self.generated = dec_gen_out[-1]
				with tf.name_scope("cost"):
					with tf.name_scope("mse"):
						mse = 0.5 * tf.reduce_sum(
								tf.square(dec_train_out[-1] - self.inputs), 1)
						tf.summary.scalar('mse', tf.reduce_mean(mse))
					with tf.name_scope("kl_divergence"):
						"""
						kl_divergence = 0.5 * (tf.reduce_sum(tf.exp(encode_logvar), 1) +
							tf.reduce_sum(tf.square(encode_means), 1) - 
							self.hidden_dim -
							tf.reduce_sum(encode_logvar, 1))
						"""
						kl_divergence = -0.5 * tf.reduce_sum(1 + encode_logvar 
                                           - tf.square(encode_means) 
                                           - tf.exp(encode_logvar), 1)
						tf.summary.scalar('kl_divergence', tf.reduce_mean(kl_divergence))
					cost = tf.reduce_mean(mse + kl_divergence)
					tf.summary.scalar('cost', cost)
				with tf.name_scope("train"):
					self.train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cost)
			self.summaries = tf.summary.merge_all()

	def train(self, inputs, epoch_num):
		file_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
		latent_samples = np.random.randn(inputs.shape[0], self.encode_architecture[-1])
		_, summaries = self.sess.run([self.train_step, self.summaries], 
					feed_dict={self.inputs : inputs, self.latent_samples : latent_samples})
		file_writer.add_summary(summaries, epoch_num)
