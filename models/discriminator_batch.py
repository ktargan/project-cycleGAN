import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa



class ConvoBlock(tf.keras.layers.Layer):
  '''Convolutional Block that combines a Convlayer with BatchNorm and leakyReLU

  Keyword Arguments: Arguments needed for Convolutional layer'''
  def __init__(self,nr_filters, strides, kernel_initializer):
    super(ConvoBlock, self).__init__()

    self.conv = tf.keras.layers.Conv2D(filters=nr_filters, kernel_size = 4, strides=strides, padding = 'same',
                           kernel_initializer = kernel_initializer)

    self.norm_layer = tf.keras.layers.BatchNormalization()

    #Zhu et al. use leaky ReLUs with a slope of 0.2.
    self.activation = tf.keras.layers.LeakyReLU(0.2)

  def call(self, x, training):
    x = self.conv(x)
    x = self.norm_layer(x, training)
    x = self.activation(x)

    return x


class Discriminator(tf.keras.Model):
  '''Discrimintaor: a PatchGAN Discriminator with 4 Downsampling layers

  architecture based on Zhu et al.,

  Keyword Arguments:
   patches - the size of the image patches the discriminator works on'''
  def __init__(self, patches):
    super(Discriminator, self).__init__()

    self.patch = patches
    #70 x70 PatchGan
    #Zhu et al: C64-C128-C256-C512
    #C64: 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with 64 filters and stride 2.

    #Weights are initialized from a Gaussian distribution N(0, 0.02).
    kernel_initializer = tf.random_normal_initializer(stddev=0.02)

    #Zhu et al. do not use InstanceNorm for the first C64 layer.
    self.conv = tf.keras.layers.Conv2D(filters=64, kernel_size = 4, strides=2, padding = 'same',
                           kernel_initializer = kernel_initializer)
    #Zhu et al. use leaky ReLUs with a slope of 0.2.
    self.activation = tf.keras.layers.LeakyReLU(0.2)

    self.layering = [
      #second layer 128 filters
      ConvoBlock(128, 2, kernel_initializer),
      #second layer 256 filters
      ConvoBlock(256, 2, kernel_initializer),
      #second layer 512 filters
      ConvoBlock(512, 2, kernel_initializer) # or strides=1 different implementations around
    ]

    #final layer
    #After the last layer, they apply a convolution to produce a 1-dimensional output.
    #in most implementations: no activation in final layer (else sigmoid)
    self.final_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides = 1,padding = 'same',
                          kernel_initializer=kernel_initializer#, activation = 'sigmoid'
                          )

  def call(self,x, training):
    #patch into pieces of 70*70
    x = tf.image.random_crop(x,[1, self.patch, self.patch,3])
    x = self.conv(x)
    x = self.activation(x)

    for layer in self.layering:
      x = layer(x, training)

    x = self.final_layer(x)

    return x
