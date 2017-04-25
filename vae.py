""" Reusable network components """

import tensorflow as tf
import numpy as np
from nn_basics import ff_network
from nn_basics import feedforward
from nn_basics import generate

class VAE:
	def __init__(self, encode_architecture, gen_architecture, name="VAE"):
		if ((encode_architecture[-1] != gen_architecture[0]) or (encode_architecture[0] != gen_architecture[-1])):
			raise Exception("Bad architecture")
		self.encode_architecture = encode_architecture
		self.gen_architecture = gen_architecture
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
						self.latent_samples = tf.placeholder(tf.float32, [None, self.hidden_dim])
				# ENCODER 
				encode_hidden = ff_network(self.inputs, encode_architecture[0:-1], "E_hidden")
				encode_means = feedforward(encode_hidden[-1], [encode_architecture[-2], self.hidden_dim],
									"encode_means")
				encode_logvar = feedforward(encode_hidden[-1], [encode_architecture[-2], self.hidden_dim],
									"encode_logvar")
				# LATENT
				with tf.name_scope("latent_training"):
					latent_training = tf.multiply(self.latent_samples, tf.sqrt(tf.exp(encode_logvar))) + encode_means
				# DECODER
				with tf.name_scope("D_train"):
					dec_train_out = ff_network(latent_training, gen_architecture, "D")
				with tf.name_scope("D_gen"):
					dec_gen_out = ff_network(self.latent_samples, gen_architecture, "D", reuse_vars=True)
				self.generated = dec_gen_out[-1]
				with tf.name_scope("cost"):
					with tf.name_scope("mse"):
						mse = 0.5 * tf.reduce_mean(
							tf.reduce_sum(
								tf.square(dec_train_out[-1] - self.inputs), 1)
						)
						tf.summary.scalar('mse', mse)
					with tf.name_scope("kl_divergence"):
						kl_divergence = tf.reduce_mean(
							0.5 * (tf.reduce_sum(tf.exp(encode_logvar), 1) +
							tf.matmul(encode_means, tf.transpose(encode_means)) -
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
						feed_dict={self.inputs : inputs, self.latent_samples : latent_samples})
			file_writer.add_summary(summaries, epoch_num)	
			saver.save(sess, self.save_dir)
	
	def generate(self):
		generate(self)
	
	def __str__(self):
		with self.graph.as_default():
			return "\n".join([var.name for var in tf.trainable_variables()])