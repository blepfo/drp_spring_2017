""" GAN """

import tensorflow as tf
import numpy as np
from gen_model import Gen_Model
from nn_basics import ff_network

epsilon = 1e-3

class GAN(Gen_Model):
	def __init__(self, gen_architecture, dec_architecture, name="GAN"):
		if (gen_architecture[-1] != dec_architecture[0]):
			raise Exception("%d %d" % (gen_architecture[-1], dec_architecture[0]))
		super().__init__(name)
		self.gen_architecture = gen_architecture
		self.dec_architecture = dec_architecture
		self.graph = tf.Graph()
		self.hidden_dim = gen_architecture[0]
		# Save model to ./saves/<name>
		self.save_dir = "".join(["./saves/", name])
		# Log model to ./logs/<name>
		self.log_dir = "".join(["./logs/", name])
		with self.graph.as_default():
			with tf.variable_scope(name):
				with tf.name_scope("inputs"):
					with tf.name_scope("latent_sample"):
						self.latent_samples = tf.placeholder(tf.float32, [None, gen_architecture[0]])
						tf.summary.histogram('latent_samples', self.latent_samples)
					with tf.name_scope("input_image"):
						self.input_image = tf.placeholder(tf.float32, [None, dec_architecture[0]])
				# GENERATOR NETWORK
				gen_network = ff_network(gen_architecture, "G")
				gen_out = gen_network.compute_output(self.latent_samples)
				self.generated = gen_out[-1]
				# DISCRIMINATOR NETWORK
				# Make sure discriminator has one output node
				if (dec_architecture[-1] != 1):
					dec_architecture.append(1)
				dec_activ_funcs = ([tf.nn.relu] * (len(dec_architecture) - 1)).append(tf.sigmoid)
				dec_network = ff_network(dec_architecture, "D", dec_activ_funcs)
				with tf.name_scope("D_in"):
					dec_in_out = dec_network.compute_output(self.input_image)
					tf.summary.scalar('D_in_out', tf.reduce_mean(dec_in_out[-1]))
				with tf.name_scope("D_gen"):
					dec_gen_out = dec_network.compute_output(self.generated)	
					tf.summary.scalar('D_gen_out', tf.reduce_mean(dec_gen_out[-1]))
				# TRAINING DETAILS
				# Add epslon inside tf.log() to prevent log(0)
				with tf.name_scope("cost"):
					with tf.name_scope("D_cost"):
						dec_cost = -0.5 * tf.reduce_mean(tf.log((dec_in_out[-1] + epsilon)) + tf.log((1 - dec_gen_out[-1] + epsilon)))
						tf.summary.scalar('D_cost', dec_cost)
					with tf.name_scope("G_cost"):
						gen_cost = -0.5 * tf.reduce_mean(tf.log(dec_gen_out[-1]+epsilon))
						tf.summary.scalar('G_cost', gen_cost)
				# Only train correct subset of parameters
					self.gen_params = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="".join([self.name, "/G"]))
					self.dec_params = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="".join([self.name, "/D"]))
				with tf.name_scope("train"):
					self.train_gen = tf.train.AdamOptimizer(1e-3).minimize(gen_cost, var_list=self.gen_params)
					self.train_dec = tf.train.AdamOptimizer(1e-3).minimize(dec_cost, var_list=self.dec_params)
				self.merged = tf.summary.merge_all()
				
	def train(self, inputs, epoch_num, log=False):
		# Create separate latent inputs for training discriminator and generator
		latent_dec = np.random.randn(inputs.shape[0], self.gen_architecture[0])
		latent_gen = np.random.randn(inputs.shape[0], self.gen_architecture[0])
		# Train discriminator
		self.sess.run(self.train_dec, 
			feed_dict={self.input_image : inputs, self.latent_samples : latent_dec})
		# Train generator
		self.sess.run(self.train_gen, 
			feed_dict={self.latent_samples : latent_gen})
		if log:
			file_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
			# Get summaries for current batch
			summary = self.sess.run(self.merged,
				feed_dict={self.input_image : inputs, self.latent_samples : latent_dec})
			file_writer.add_summary(summary, epoch_num)