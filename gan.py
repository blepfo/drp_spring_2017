""" GAN """

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from nn_basics import ff_network

epsilon = 1e-3

class GAN:
	def __init__(self, gen_architecture, dec_architecture, name="GAN"):
		if (gen_architecture[-1] != dec_architecture[0]):
			raise Exception("%d %d" % (gen_architecture[-1], dec_architecture[0]))
		self.gen_architecture = gen_architecture
		self.dec_architecture = dec_architecture
		self.name = name
		self.graph = tf.Graph()
		# Save model to ./saves/<name>
		self.save_dir = "".join(["./saves/", name])
		# Log model to ./logs/<name>
		self.log_dir = "".join(["./logs/", name])
		with self.graph.as_default():
			with tf.variable_scope(name):
				with tf.name_scope("inputs"):
					with tf.name_scope("latent_sample"):
						self.latent_sample = tf.placeholder(tf.float32, [None, gen_architecture[0]])
					with tf.name_scope("input_image"):
						self.input_image = tf.placeholder(tf.float32, [None, dec_architecture[0]])
				# GENERATOR NETWORK
				gen_layers, gen_out = ff_network(gen_architecture, self.latent_sample, "G")
				self.generated = gen_out[-1]
				# DISCRIMINATOR NETWORK
				# Make sure discriminator has one output node
				if (dec_architecture[-1] != 1):
					dec_architecture.append(1)
				dec_activ_funcs = ([tf.nn.relu] * (len(dec_architecture) - 1)).append(tf.sigmoid)
				with tf.name_scope("D_in"):
					dec_in_layers, dec_in_out = ff_network(dec_architecture, self.input_image, "D",
								activation_funcs=dec_activ_funcs)
				with tf.name_scope("D_G"):
					dec_gen_layers, dec_gen_out = ff_network(dec_architecture, gen_out[-1], "D", 
								activation_funcs=dec_activ_funcs, reuse_vars=True)
				# TRAINING DETAILS
				# Add epslon inside tf.log() to prevent log(0)
				with tf.name_scope("cost"):
					with tf.name_scope("D_cost"):
						dec_cost = -0.5 * tf.reduce_mean(tf.log(tf.maximum(dec_in_out[-1], epsilon)) + tf.log(tf.maximum(1 - dec_gen_out[-1], epsilon)))
					with tf.name_scope("G_cost"):
						gen_cost = -0.5 * tf.reduce_mean(tf.log(dec_gen_out[-1]+epsilon))
				# Only train correct subset of parameters
					self.gen_params = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="".join([self.name, "/G"]))
					self.dec_params = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="".join([self.name, "/D"]))
				with tf.name_scope("train"):
					self.train_gen = tf.train.AdamOptimizer(1e-3).minimize(gen_cost, var_list=self.gen_params)
					self.train_dec = tf.train.AdamOptimizer(1e-3).minimize(dec_cost, var_list=self.dec_params)
				# SUMMARIES
				with tf.name_scope('summaries'):
					tf.summary.histogram('latent_samples', self.latent_sample)
					tf.summary.scalar('D_in_out', tf.reduce_mean(dec_in_out[-1]))
					tf.summary.scalar('D_gen_out', tf.reduce_mean(dec_gen_out[-1]))
					tf.summary.scalar('D_cost', dec_cost)
					tf.summary.scalar('G_cost', gen_cost)
					self.merged = tf.summary.merge_all()
		# Initialize model and save
		with tf.Session(graph=self.graph) as sess:
			sess.run(tf.global_variables_initializer())
			tf.train.Saver().save(sess, self.save_dir)
	
	def train(self, inputs, epoch_num):
		with tf.Session(graph=self.graph) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, self.save_dir)
			file_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
			# Create separate latent inputs for training discriminator and generator
			latent_dec = np.random.randn(inputs.shape[0], self.gen_architecture[0])
			latent_gen = np.random.randn(inputs.shape[0], self.gen_architecture[0])
			# Train discriminator
			sess.run(self.train_dec, 
				feed_dict={self.input_image : inputs, self.latent_sample : latent_dec})
			# Train generator
			sess.run(self.train_gen, 
				feed_dict={self.latent_sample : latent_gen})
			# Get summaries for current batch
			summary = sess.run(self.merged,
				feed_dict={self.input_image : inputs, self.latent_sample : latent_dec})
			file_writer.add_summary(summary, epoch_num)
			
			if (epoch_num % 5 == 0):
				latent = np.random.randn(1, self.gen_architecture[0])
				image = sess.run(self.generated, feed_dict={self.latent_sample : latent})
				plt.imshow(np.reshape(image, [28,28]), cmap="Greys")
				plt.axis('off')
				plt.show()

			saver.save(sess, self.save_dir)
				
	def generate(self):
		with tf.Session(graph=self.graph) as sess:
			tf.train.Saver().restore(sess, self.save_dir)
			latent = np.random.randn(1, self.gen_architecture[0])
			image = sess.run(self.generated, feed_dict={self.latent_sample : latent})
			plt.imshow(np.reshape(image, [28,28]), cmap="Greys")
			plt.axis('off')
			plt.show()
	
	def __str__(self):
		""" String of all trainable parameters in the network """
		with self.graph.as_default():
			return "\n".join([var.name for var in tf.trainable_variables()])	