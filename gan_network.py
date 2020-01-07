#!/usr/bin/env python
# coding: utf-8

# 
# Following code work for image to image translation with two generator GAN.
# Baseline Code is designed by Tensorflow and it is customised by Omkar Thawakar
# as a part of research project.

# ## Import TensorFlow and other libraries

from __future__ import absolute_import, division, print_function, unicode_literals

try:
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.system('color 1')
import time
import sys
from absl import app

from matplotlib import pyplot as plt
from IPython import display

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda, Add, ReLU, MaxPooling2D
import datetime
import numpy as np
from termcolor import colored, cprint

experiment = 'Experiment/'
PATH = 'dataset/'
LAMBDA = 100
epochs = 100
loop = 7
nf=8

restore_checkpoint = None

BUFFER_SIZE = 1
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = ( input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 256, 256)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file):
	input_image, real_image = load(image_file)
	#input_image, real_image = random_jitter(input_image, real_image)
	input_image, real_image = normalize(input_image, real_image)

	return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

OUTPUT_CHANNELS = 3

def batch_norm(tensor):
    return tf.keras.layers.BatchNormalization(axis=3,epsilon=1e-5, momentum=0.1, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor)

##################################### Generator 1 definition ##################################################

class Generator_1:
	def __init__(self):
	    self.inputs = tf.keras.layers.Input(shape=[256,256,3])
	    self.name = 'Generator_1/'
	    print(colored('='*50,'green'))
	    print('input shape ::: ',self.inputs.shape)
	    self.generator = self.build_generator()
	    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	def upsample_block(self, filters, size, apply_dropout=False, name='upsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
		                                padding='same',
		                                kernel_initializer=initializer,
		                                use_bias=False,
		                                name = self.name+name))
		result.add(tf.keras.layers.BatchNormalization())
		if apply_dropout:
			result.add(tf.keras.layers.Dropout(0.5))
		result.add(tf.keras.layers.ReLU())
		return result

	def downsample_block(self,filters=16, size=3, apply_batchnorm=True, name='downsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
		                         kernel_initializer=initializer, use_bias=False,
		                        name='Conv_'+name))
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def Convolution(self,filters, size, apply_batchnorm=True, name='convolution'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
		                         kernel_initializer=initializer, use_bias=True, name=self.name+name))
		if apply_batchnorm:
			result.add(tf.keras.layers.BatchNormalization())
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def generator_loss(self, disc_generated_output, gen_output, target):
		gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
		# mean absolute error
		l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
		total_gen_loss = gan_loss + (LAMBDA * l1_loss)
		return total_gen_loss, gan_loss, l1_loss

	def build_generator(self):
	    print(colored('################## Build generator 1 ##################','green'))

	    down1 = self.downsample_block(filters=16, size=3, name='DownSample1_')(self.inputs)
	    print(colored('down1 ::: {}'.format(down1.shape),'green'))

	    down2 = self.downsample_block(filters=16,size=3, name='DownSample1_')(down1)
	    print(colored('down2 ::: {}'.format(down2.shape),'green'))

	    up1 = self.upsample_block(filters=16,size=3, name='UpSample1_')(down2)
	    print(colored('up1 ::: {}'.format(up1.shape),'green'))
	    tensor = tf.keras.layers.Concatenate()([up1, down1])

	    up2 = self.upsample_block(filters=16,size=3, name='UpSample1_')(tensor)
	    print(colored('up2 ::: {}'.format(up2.shape),'green'))

	    tensor = self.Convolution(filters=3, size=3, name=self.name+'output')(up2)

	    print(colored('output ::: {}'.format(tensor.shape),'green'))
	    print(colored('='*50,'green'))
	    
	    return tf.keras.Model(inputs=self.inputs, outputs=tensor)

class Discriminator1():

	def __init__(self):
		self.inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
		self.tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
		self.input_ = tf.keras.layers.concatenate([self.inp, self.tar],axis=3)
		self.name = 'Disriminator1/'
		self.discriminator = self.build_discriminator()

	def conv2d(self, filters, size,stride=2,name='conv2d'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(tf.keras.layers.Conv2D(filters, size, strides=stride,
		                                padding='same',
		                                kernel_initializer=initializer,
		                                use_bias=False,
		                                name = self.name+name))
		return result

	def downsample_block(self,filters=16, size=3, apply_batchnorm=True, name='downsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
		                         kernel_initializer=initializer, use_bias=False,
		                        name='Conv_'+name))
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def batch_norm(self, tensor):
		return tf.keras.layers.BatchNormalization(axis=3,epsilon=1e-5, momentum=0.1, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor)

	def discriminator_loss(self, disc_real_output, disc_generated_output):
		loss = tf.reduce_mean(-(tf.math.log(disc_real_output + 1e-12) + tf.math.log(1 - disc_generated_output + 1e-12)))
		return loss

	def build_discriminator(self):
		initializer = tf.random_normal_initializer(0., 0.02)
		print(colored('################## Build Discriminator 1 ##################','yellow'))

		down1 = self.downsample_block(filters=16, size=4, name='DownSample1_')(self.input_)
		print(colored('down1 ::: {}'.format(down1.shape),'yellow'))

		down2 = self.downsample_block(filters=16, size=4, name='DownSample2_')(down1)
		print(colored('down2 ::: {}'.format(down2.shape),'yellow'))

		down3 = self.downsample_block(filters=16, size=4, name='DownSample3_')(down2)
		print(colored('down3 ::: {}'.format(down3.shape),'yellow'))

		zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 
		conv = tf.keras.layers.Conv2D(512, 4, strides=1,
		                            kernel_initializer=initializer,
		                            use_bias=False)(zero_pad1) 

		batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

		leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

		zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 

		last = tf.keras.layers.Conv2D(1, 4, strides=1,
		                            kernel_initializer=initializer)(zero_pad2) 

		return tf.keras.Model(inputs=[self.inp, self.tar], outputs=last) 

###############################################################################################################

##################################### Generator 2 definition ##################################################

class Generator_2:
	def __init__(self):
	    self.inputs = tf.keras.layers.Input(shape=[256,256,3])
	    self.name = 'Generator_1/'
	    print(colored('='*50,'green'))
	    print('input shape ::: ',self.inputs.shape)
	    self.generator = self.build_generator()  ### creating generator model
	    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	def upsample_block(self, filters, size, apply_dropout=False, name='upsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
		                                padding='same',
		                                kernel_initializer=initializer,
		                                use_bias=False,
		                                name = self.name+name))
		result.add(tf.keras.layers.BatchNormalization())
		if apply_dropout:
			result.add(tf.keras.layers.Dropout(0.5))
		result.add(tf.keras.layers.ReLU())
		return result

	def downsample_block(self,filters=16, size=3, apply_batchnorm=True, name='downsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
		                         kernel_initializer=initializer, use_bias=False,
		                        name='Conv_'+name))
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def Convolution(self,filters, size, apply_batchnorm=True, name='convolution'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
		                         kernel_initializer=initializer, use_bias=True, name=self.name+name))
		if apply_batchnorm:
			result.add(tf.keras.layers.BatchNormalization())
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def generator_loss(self, disc_generated_output, gen_output, target):
		gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
		# mean absolute error
		l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
		total_gen_loss = gan_loss + (LAMBDA * l1_loss)
		return total_gen_loss, gan_loss, l1_loss

	def build_generator(self):
	    print(colored('################## Build generator 1 ##################','cyan'))

	    down1 = self.downsample_block(filters=16, size=3, name='DownSample1_')(self.inputs)
	    print(colored('down1 ::: {}'.format(down1.shape),'cyan'))

	    down2 = self.downsample_block(filters=16,size=3, name='DownSample2_')(down1)
	    print(colored('down2 ::: {}'.format(down2.shape),'cyan'))

	    up1 = self.upsample_block(filters=16,size=3, name='UpSample1_')(down2)
	    print(colored('up1 ::: {}'.format(up1.shape),'cyan'))
	    tensor = tf.keras.layers.Concatenate()([up1, down1])

	    up2 = self.upsample_block(filters=16,size=3, name='UpSample2_')(tensor)
	    print(colored('up2 ::: {}'.format(up2.shape),'cyan'))

	    tensor = self.Convolution(filters=3, size=3, name=self.name+'output')(up2)

	    print(colored('output ::: {}'.format(tensor.shape),'cyan'))
	    print(colored('='*50,'cyan'))
	    
	    return tf.keras.Model(inputs=self.inputs, outputs=tensor)

class Discriminator2():

	def __init__(self):
		self.inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
		self.tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
		self.input_ = tf.keras.layers.concatenate([self.inp, self.tar],axis=3)
		self.name = 'Disriminator1/'
		self.discriminator = self.build_discriminator()

	def conv2d(self, filters, size,stride=2,name='conv2d'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(tf.keras.layers.Conv2D(filters, size, strides=stride,
		                                padding='same',
		                                kernel_initializer=initializer,
		                                use_bias=False,
		                                name = self.name+name))
		return result

	def downsample_block(self,filters=16, size=3, apply_batchnorm=True, name='downsample'):
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.keras.Sequential()
		result.add(
		  tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
		                         kernel_initializer=initializer, use_bias=False,
		                        name='Conv_'+name))
		result.add(tf.keras.layers.LeakyReLU())
		return result

	def batch_norm(self, tensor):
		return tf.keras.layers.BatchNormalization(axis=3,epsilon=1e-5, momentum=0.1, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor)

	def discriminator_loss(self, disc_real_output, disc_generated_output):
		loss = tf.reduce_mean(-(tf.math.log(disc_real_output + 1e-12) + tf.math.log(1 - disc_generated_output + 1e-12)))
		return loss

	def build_discriminator(self):
		initializer = tf.random_normal_initializer(0., 0.02)
		print(colored('################## Build Discriminator 1 ##################','magenta'))

		down1 = self.downsample_block(filters=16, size=4, name='DownSample1_')(self.input_)
		print(colored('down1 ::: {}'.format(down1.shape),'magenta'))

		down2 = self.downsample_block(filters=16, size=4, name='DownSample2_')(down1)
		print(colored('down2 ::: {}'.format(down2.shape),'magenta'))

		down3 = self.downsample_block(filters=16, size=4, name='DownSample3_')(down2)
		print(colored('down3 ::: {}'.format(down3.shape),'magenta'))

		zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 
		conv = tf.keras.layers.Conv2D(512, 4, strides=1,
		                            kernel_initializer=initializer,
		                            use_bias=False)(zero_pad1) 

		batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

		leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

		zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 

		last = tf.keras.layers.Conv2D(1, 4, strides=1,
		                            kernel_initializer=initializer)(zero_pad2) 

		return tf.keras.Model(inputs=[self.inp, self.tar], outputs=last) 

class GAN(object):
	"""GAN class.
	Args:
	epochs: Number of epochs.
	enable_function: If true, train step is decorated with tf.function.
	buffer_size: Shuffle buffer size..
	batch_size: Batch size.
	"""
	def __init__(self, epochs,path,mode):
		self.epochs = epochs
		self.path = 'dataset'
		self.lambda_value = 100
		self.gen1 = Generator_1()
		self.generator1 = self.gen1.generator
		self.print_info(self.generator1, 'Generator 1')

		self.disc1 = Discriminator1()
		self.discriminator1 = self.disc1.discriminator
		self.print_info(self.discriminator1, 'discriminator 1')

		self.disc2 = Discriminator2()
		self.discriminator2 = self.disc2.discriminator
		self.print_info(self.discriminator2, 'discriminator 2')

		self.gen2 = Generator_2()
		self.generator2 = self.gen2.generator
		self.print_info(self.generator2, 'Generator 2')

		self.generator1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		self.discriminator1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

		self.generator2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		self.discriminator2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

		self.checkpoint_dir1 = self.path + './training_checkpoints/' + 'gen1'
		self.checkpoint_prefix1 = os.path.join(self.checkpoint_dir1, "ckpt")
		self.checkpoint1 = tf.train.Checkpoint(generator1_optimizer=self.generator1_optimizer,
		                                 discriminator1_optimizer=self.discriminator1_optimizer,
		                                 generator1=self.generator1,
		                                 discriminator1=self.discriminator1,
		                                 )

		self.checkpoint_dir2 = self.path + './training_checkpoints' + 'gen2'
		self.checkpoint_prefix2 = os.path.join(self.checkpoint_dir2, "ckpt")
		self.checkpoint2 = tf.train.Checkpoint(generator2_optimizer=self.generator2_optimizer,
		                                 discriminator2_optimizer=self.discriminator2_optimizer,
		                                 generator2=self.generator2,
		                                 discriminator2=self.discriminator2)

		log_dir = self.path + "logs/"
		self.summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

	def generate_images(self, test_input, tar, number, folder = experiment, mode='train'):
		if mode == 'train':
		    gen1_prediction = self.generator1(test_input, training=True)
		    gen2_prediction = self.generator2(gen1_prediction, training=True)
		    display_list = [test_input[0], gen1_prediction[0], gen2_prediction[0], tar[0]]
		    image = np.hstack([img for img in display_list])
		    try :
		        os.mkdir(folder+'{}'.format(mode))
		    except:
		        pass
		    plt.imsave(folder+'{}/{}_.png'.format(mode,number), np.array((image * 0.5 + 0.5)*255, dtype='uint8'))
		elif mode == 'test' :
		    gen1_prediction = self.generator1(test_input, training=True)
		    gen2_prediction = self.generator2(gen1_prediction, training=True)
		    display_list = [test_input[0], gen1_prediction[0], gen2_prediction[0], tar[0]]
		    image = np.hstack([img for img in display_list])
		    try :
		        os.mkdir(folder+'{}'.format(mode))
		    except:
		        pass
		    plt.imsave(folder+'{}/{}_.png'.format(mode,umber), np.array((image * 0.5 + 0.5)*255, dtype='uint8'))
		else:
		    print('Enter valid mode eighter [!]train or [!]test')
		    exit(0)

	def print_info(self,object, name):
		print('='*50)
		text = colored('Total Trainable parameters of {} are :: {}'.format(object.count_params(), name), 'red', attrs=['reverse','blink'])
		print(text)
		print('='*50)

	def train_step(self,input_image, target, epoch):
	    with tf.GradientTape() as gen_tape1, tf.GradientTape() as disc_tape1, tf.GradientTape() as gen_tape2, tf.GradientTape() as disc_tape2 :
	        gen1_output = self.generator1(input_image, training=True)
	        disc1_real_output = self.discriminator1([input_image, target], training=True)
	        disc1_generated_output = self.discriminator1([input_image, gen1_output], training=True)
	        gen1_total_loss, gen1_gan_loss, gen1_l1_loss = self.gen1.generator_loss(disc1_generated_output, gen1_output, target)
	        disc1_loss = self.disc1.discriminator_loss(disc1_real_output, disc1_generated_output)

	        gen2_output = self.generator2(gen1_output, training=True)
	        disc2_real_output = self.discriminator2([input_image, target], training=True)
	        disc2_generated_output = self.discriminator2([input_image, gen2_output], training=True)
	        gen2_total_loss, gen2_gan_loss, gen2_l1_loss = self.gen2.generator_loss(disc2_generated_output, gen2_output, target)
	        disc2_loss = self.disc2.discriminator_loss(disc2_real_output, disc2_generated_output)

	    ######################### Generator 1 Gradients ############################
	    generator1_gradients = gen_tape1.gradient(gen1_total_loss,
	                                          self.generator1.trainable_variables)
	    discriminator1_gradients = disc_tape1.gradient(disc1_loss,
	                                               self.discriminator1.trainable_variables)
	    self.generator1_optimizer.apply_gradients(zip(generator1_gradients,
	                                          self.generator1.trainable_variables))
	    self.discriminator1_optimizer.apply_gradients(zip(discriminator1_gradients,
	                                              self.discriminator1.trainable_variables))
	    ######################### Generator 2 Gradients ############################
	    generator2_gradients = gen_tape2.gradient(gen2_total_loss,
	                                          self.generator2.trainable_variables)
	    discriminator2_gradients = disc_tape2.gradient(disc2_loss,
	                                               self.discriminator2.trainable_variables)
	    self.generator2_optimizer.apply_gradients(zip(generator2_gradients,
	                                          self.generator2.trainable_variables))
	    self.discriminator2_optimizer.apply_gradients(zip(discriminator2_gradients,
	                                              self.discriminator2.trainable_variables))

	    with self.summary_writer.as_default():
	        tf.summary.scalar('gen1_total_loss', gen1_total_loss, step=epoch)
	        tf.summary.scalar('gen1_gan_loss', gen1_gan_loss, step=epoch)
	        tf.summary.scalar('gen1_l1_loss', gen1_l1_loss, step=epoch)
	        tf.summary.scalar('disc1_loss', disc1_loss, step=epoch)
	        tf.summary.scalar('gen2_total_loss', gen2_total_loss, step=epoch)
	        tf.summary.scalar('gen2_gan_loss', gen2_gan_loss, step=epoch)
	        tf.summary.scalar('gen2_l1_loss', gen2_l1_loss, step=epoch)
	        tf.summary.scalar('disc2_loss', disc2_loss, step=epoch)

	    outputs = {
	                'gen1_total_loss' : gen1_total_loss , 
	                'gen1_gan_loss' : gen1_gan_loss, 
	                'gen1_l1_loss' : gen1_l1_loss,
	                'disc1_loss' : disc1_loss,
	                'gen2_total_loss' : gen2_total_loss , 
	                'gen2_gan_loss' : gen2_gan_loss, 
	                'gen2_l1_loss' : gen2_l1_loss,
	                'disc2_loss' : disc2_loss, 
	            }

	    return outputs

	def fit(self, train_ds, epochs, test_ds):
	    for epoch in range(self.epochs):
	        start = time.time()
	        display.clear_output(wait=True)
	        for example_input, example_target in test_ds.take(1):
	            self.generate_images(example_input, example_target, epoch)
	        print(colored("Epoch: {}".format(epoch),'green',attrs=['reverse','blink']))

	        # Train
	        for n, (input_image, target) in train_ds.enumerate():
	            print('.', end='')
	            if (n+1) % 100 == 0:
	                print()
	            outputs = self.train_step(input_image, target, epoch)
	        print()
	        print('='*50)
	        print(colored('[!] gen1_total_loss :: {}'.format(outputs['gen1_total_loss']),'green'))
	        print(colored('[!] gen1_gan_loss :: {}'.format(outputs['gen1_gan_loss']),'green'))
	        print(colored('[!] gen1_l1_loss :: {}'.format(outputs['gen1_l1_loss']),'green'))
	        print(colored('[!] disc1_loss :: {}'.format(outputs['disc1_loss']),'green'))

	        print(colored('[!] gen2_total_loss :: {}'.format(outputs['gen2_total_loss']),'yellow'))
	        print(colored('[!] gen2_gan_loss :: {}'.format(outputs['gen2_gan_loss']),'yellow'))
	        print(colored('[!] gen2_l1_loss :: {}'.format(outputs['gen2_l1_loss']),'yellow'))
	        print(colored('[!] disc2_loss :: {}'.format(outputs['disc2_loss']),'yellow'))
	        print('='*50)

	        # saving (checkpoint) the model every 20 epochs
	        if (epoch + 1) % 5 == 0:
	            checkpoint1.save(file_prefix = checkpoint_prefix1)
	            checkpoint2.save(file_prefix = checkpoint_prefix2)

	        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
	                                                time.time()-start))
	    checkpoint1.save(file_prefix = checkpoint_prefix1)
	    checkpoint2.save(file_prefix = checkpoint_prefix2)


def run_main(argv):
  del argv
  kwargs = {'epochs': 100, 'path': 'dataset/',
  			'mode':'train',
            }
  main(**kwargs)


def main(epochs, path,mode ):

	gan_object = GAN(epochs,path,mode)
	print ('Training ...')
	if mode=='train':
		############# train dataset #################
		train_dataset = tf.data.Dataset.list_files(path+'train/*.jpg')
		train_dataset = train_dataset.map(load_image_train,
		                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
		train_dataset = train_dataset.shuffle(1)
		train_dataset = train_dataset.batch(1)
		############# test dataset ##################
		test_dataset = tf.data.Dataset.list_files(path+'test/*.jpg')
		test_dataset = test_dataset.map(load_image_test)
		test_dataset = test_dataset.batch(1)
		print('Training !!!!!')
		gan_object.fit(train_dataset,epochs, test_dataset)

	elif mode=='test':
		test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
		test_dataset = test_dataset.map(load_image_test)
		test_dataset = test_dataset.batch(BATCH_SIZE)



if __name__ == '__main__':
  app.run(run_main)
