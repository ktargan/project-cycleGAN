import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import layers


class ResidualBlock(tf.keras.layers.Layer):
  '''Defines a Block as used in the ResNet architecture
  2 covolutional layers
  InstanceNormalization (instead of Batchnorm)
  Reflection Padding

  Keyword Arguments:
  nr_filters : amount of filters for the convolutional layers
  kernel_initializer: how to inititialize the weights'''
  def __init__(self, nr_filters, kernel_initializer, size):
    super(ResidualBlock,self).__init__()

    #use padding to keep the featuremap size constant even after applying convolutions
    #self.padd1 = layers.ReflectionPadding2D()

    #3x3 conv, normalization, relu, 3x3 conv, normalization, relu
    self.conv_1 = tf.keras.layers.Conv2D(filters = nr_filters, kernel_size = 3,
                                         kernel_initializer = kernel_initializer
                                        #kernel_regularizer = tf.keras.regularizers.l2(0.01),
                                         #padding = 'same'
                                         )

    #instance normalization
    self.norm_1 = tfa.layers.InstanceNormalization(
          gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

    self.relu_1 = tf.keras.layers.ReLU()

    #self.padd2 = layers.ReflectionPadding2D()
    self.conv_2 = tf.keras.layers.Conv2D(filters = nr_filters, kernel_size = 3,
                                         kernel_initializer = kernel_initializer
                                        # padding = 'same'
                                         )
    self.norm_2 = tfa.layers.InstanceNormalization(
            gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

    self.crop = tf.keras.layers.experimental.preprocessing.CenterCrop(height=size, width = size)


    self.relu_2 = tf.keras.layers.ReLU()

  def call(self, start_x):
    #x = self.padd1(start_x)
    x = self.conv_1(start_x)
    x = self.norm_1(x)

    x = self.relu_1(x)
    #x = self.padd2(x)
    x = self.conv_2(x)
    x = self.norm_2(x)
    # skip connections are used: add up the input of the block to its output
    # but first the input needs to be cropped as the output is smaller in max_size
    # (Johnson et al. do not use padding in residual blocks)
    start_x = self.crop(start_x)
    x = x + start_x
    x = self.relu_2(x)
    return x



class DownsampleBlock(tf.keras.layers.Layer):
  '''Block used for downsampling images/featuremaps with strided convolutions
  Convolutional layer, Instance Normalization, LeakyReLU activation

  Keyword Arguments:
  parameters needed to define convolutional layer'''
  def __init__(self, nr_filters, kernel_size, stride, padding, kernel_initializer):
    super(DownsampleBlock,self).__init__()

    self.conv = tf.keras.layers.Conv2D(nr_filters, kernel_size= kernel_size, strides = stride,
                                       padding=padding, kernel_initializer = kernel_initializer)
    self.norm_layer = tfa.layers.InstanceNormalization(
            gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

    self.activation = tf.keras.layers.LeakyReLU()

  def call(self, x):

    x = self.conv(x)
    x = self.norm_layer(x)

    x = self.activation(x)

    return x



class UpsampleBlock(tf.keras.layers.Layer):
  '''Block used for upsampling images with fractionally strided convolution,
  TransposedConvolutional layer, Instance Normalization, LeakyReLU activation

  Keyword Arguments:
  parameters needed to define convolutional layer'''
  def __init__(self, nr_filters, kernel_size, stride, kernel_initializer):
    super(UpsampleBlock,self).__init__()

    self.conv = tf.keras.layers.Conv2DTranspose(nr_filters, kernel_size= kernel_size, strides = stride,
                                       padding='same', kernel_initializer = kernel_initializer)
    self.norm_layer = tfa.layers.InstanceNormalization(
            gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

    self.activation = tf.keras.layers.LeakyReLU()

  def call(self, x):

    x = self.conv(x)
    x = self.norm_layer(x)

    x = self.activation(x)

    return x




class Generator(tf.keras.Model):
  '''Generator is built up from different Down-, upsampling and Residual blocks

  architecture based Image Transformation Network by Johnson'''
  def __init__(self):
    super(Generator, self).__init__()
    #structure as referenced by Zhu et al. is different from Johnson architecture:
    #Zhu: c7s1-64,d128,d256,R256,R256,R256, R256,R256,R256,u128,u64,c7s1-3

    #Weights are initialized from a Gaussian distribution N(0, 0.02).
    kernel_initializer = tf.random_normal_initializer(stddev=0.02)

    #padd before using a 7x7 kernel as done in the first layer (Defined in strided block)
    self.padd1 = layers.ReflectionPadding2D(padding=(40, 40))

    self.strided_block = [
      #Johnson: 32 x9 x9 conv, stride 1 -> 32?? 208??208
      #but in the cycle gan paper use 7x7, 64
      DownsampleBlock(nr_filters =32, kernel_size = 9, stride = 1, padding = 'same', kernel_initializer = kernel_initializer),

      #Johnson: 64??3??3, conv, stride 2 -> 64??104?? 104
      #Zhu: 128x3x3
      DownsampleBlock(nr_filters = 64,kernel_size = 3, stride = 2, padding = 'same',kernel_initializer = kernel_initializer),

      #Johnson: 128??3??3 conv, stride 2 -> 128??52x52
      #Zhu: 256
      DownsampleBlock(nr_filters =128, kernel_size = 3, stride = 2, padding = 'same', kernel_initializer = kernel_initializer)
    ]

    #Residual blocks -> 5 in Johnson architecture
    self.blocks = [
        # -> 128 x 48 x48
      ResidualBlock(128, kernel_initializer = kernel_initializer, size = 48),
      # 128 x 44 x44
      ResidualBlock(128, kernel_initializer = kernel_initializer, size = 44),
      # 128 x 40x40
      ResidualBlock(128, kernel_initializer = kernel_initializer, size = 40),
      # 128 x 36 x 36
      ResidualBlock(128, kernel_initializer = kernel_initializer, size = 36),
      # 128 x 32 x32
      ResidualBlock(128, kernel_initializer = kernel_initializer, size = 32),
    ]

    self.transposed_block = [
      #64??3??3 conv, stride 1/2 -> 64??64 x64
      UpsampleBlock(64,3,2, kernel_initializer),
      #32??3??3 conv, stride 1/2 -> 32??128??128
      UpsampleBlock(32,3,2, kernel_initializer),
    ]
    #padding:
    #self.padd2 = layers.ReflectionPadding2D(padding=(self.padds, self.padds))

    #3??9??9 conv, stride 1 3??128??128
    #Johnson et al. use tanh
    self.final_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=9,
                                             strides =1, activation = tf.keras.activations.tanh,
                                             padding = 'same', kernel_initializer = kernel_initializer)


  def call(self, x):
    x = self.padd1(x)

    for layer in self.strided_block:
      x = layer(x)

    for block in self.blocks:
      x = block(x)

    for layer in self.transposed_block:
      x = layer(x)


    #x = self.padd2(x)

    x = self.final_layer(x)
    return x
