""" GAN """

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from nn_basics import ff_network
from nn_basics import ff_layer
from nn_basics import feedforward

class GAN:
	def __init__(self, gen_architecture, dec_architecture, name="GAN"):
		if (gen_architecture[-1] != dec_architecture[0]):
			raise Exception("%d %d" % (gen_architecture[-1], dec_architecture[0]))
		self.gen_architecture = gen_architecture
		self.dec_architecture = dec_architecture
		self.name = name
		self.graph = tf.Graph()
		with self.graph.as_default():
			with tf.variable_scope(name):
				with tf.name_scope("inputs"):
					with tf.name_scope("latent_sample"):
						self.latent_sample = tf.placeholder(tf.float32, [None, gen_architecture[0]])
						tf.summary.histogram('latent_samples', self.latent_sample)
					with tf.name_scope("input_image"):
						self.input_image = tf.placeholder(tf.float32, [None, dec_architecture[0]])
				# GENERATOR NETWORK
				gen_layers, gen_out = ff_network(gen_architecture, self.latent_sample, "G")
				self.generated = gen_out[-1]
				# DISCRIMINATOR NETWORK
				# Create decoding layers
				dec_out_layer = ff_layer([dec_architecture[-1], 1], "D", activation_func=tf.sigmoid)
				with tf.name_scope("D_in"):
					dec_in_layers, dec_in_out = ff_network(dec_architecture, self.input_image, "D")
					# Binary classification output
					dec_in_layers.append(dec_out_layer)
					dec_in_out.append(feedforward(dec_in_layers[-1], dec_in_out[-1]))
					tf.summary.scalar('D_in_out', tf.reduce_mean(dec_in_out[-1]))
				with tf.name_scope("D_G"):
					dec_gen_layers, dec_gen_out = ff_network(dec_architecture, gen_out[-1], "D", reuse_vars=True)
					# Binary classification output
					dec_gen_layers.append(dec_out_layer)
					dec_gen_out.append(feedforward(dec_gen_layers[-1], dec_gen_out[-1]))
					tf.summary.scalar('D_gen_out', tf.reduce_mean(dec_gen_out[-1]))
				# TRAINING DETAILS
				with tf.name_scope("cost"):
					with tf.name_scope("D_cost"):
						dec_cost = -0.5 * tf.reduce_mean((tf.log(dec_in_out[-1]) + tf.log(1 - dec_gen_out[-1])))
						tf.summary.scalar('D_cost', dec_cost)
					with tf.name_scope("G_cost"):
						gen_cost = -0.5 * tf.reduce_mean(tf.log(dec_gen_out[-1]))
						tf.summary.scalar('G_cost', gen_cost)
				with tf.name_scope("train"):
					self.train_dec = tf.train.AdamOptimizer(1e-3).minimize(dec_cost)
					self.train_gen = tf.train.AdamOptimizer(1e-3).minimize(gen_cost)
				self.merged = tf.summary.merge_all()
	
	def train(self, inputs, epochs):
		with tf.Session(graph=self.graph) as sess:
			sess.run(tf.global_variables_initializer())
			file_writer = tf.summary.FileWriter('./logs', sess.graph)
			for epoch in range(epochs):
				print("EPOCH: %d" % (epoch+1))
				latent = np.random.randn(inputs.shape[0], self.gen_architecture[0])
				# Randomly train D or G
				to_train = self.train_dec if (np.random.uniform(0, 1) < 0.5) else self.train_gen
				summary, _ = sess.run([self.merged, to_train], 
					feed_dict={self.input_image : inputs, self.latent_sample : latent})
				file_writer.add_summary(summary, epoch)
			saver = tf.train.Saver()
			saver.save(sess, './model')
			latent = np.random.randn(1, self.gen_architecture[0])
			image = sess.run(self.generated, feed_dict={self.latent_sample : latent})
			plt.imshow(np.reshape(image, [28,28]), cmap="Greys")
			plt.axis('off')
			plt.show()
				
	def generate(self):
		with tf.Session(graph=self.graph) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, './model')
			latent = np.random.randn(1, self.gen_architecture[0])
			image = sess.run(self.generated, feed_dict={self.latent_sample : latent})
			plt.imshow(np.reshape(image, [28,28]), cmap="Greys")
			plt.axis('off')
			plt.show()
					