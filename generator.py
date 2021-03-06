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
  def __init__(self, nr_filters, kernel_initializer):
    super(ResidualBlock,self).__init__()

    #use padding to keep the featuremap size constant even after applying convolutions
    self.padd1 = layers.ReflectionPadding2D()

    #3x3 conv, normalization, relu, 3x3 conv, instance normalization, relu
    self.conv_1 = tf.keras.layers.Conv2D(filters = nr_filters, kernel_size = 3,
                                         kernel_initializer = kernel_initializer
                                        #kernel_regularizer = tf.keras.regularizers.l2(0.01),
                                         #padding = 'same'
                                         )

    #Instancenorm normalizes the feature channels of each image of a batch seperatly along
    #its spatial dimensions. The gamma_initializer sets the initial weights of the layer
    #to a normla distribution with mean at 0 and standard deviation at 0.02.
    self.norm_1 = tfa.layers.InstanceNormalization(
          gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

    self.relu_1 = tf.keras.layers.ReLU()

    self.padd2 = layers.ReflectionPadding2D()
    self.conv_2 = tf.keras.layers.Conv2D(filters = nr_filters, kernel_size = 3,
                                         kernel_initializer = kernel_initializer
                                        # padding = 'same'
                                         )
    self.norm_2 = tfa.layers.InstanceNormalization(
            gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

    self.relu_2 = tf.keras.layers.ReLU()

  def call(self, start_x):
    x = self.padd1(start_x)
    x = self.conv_1(x)
    x = self.norm_1(x)

    x = self.relu_1(x)
    x = self.padd2(x)
    x = self.conv_2(x)
    x = self.norm_2(x)
    # skip connections are used: add up the input to block to its output
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

    #Instancenorm normalizes the feature channels of each image of a batch seperatly along
    #its spatial dimensions. The gamma_initializer sets the initial weights of the layer
    #to a normla distribution with mean at 0 and standard deviation at 0.02.
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

    #Instancenorm normalizes the feature channels of each image of a batch seperatly along
    #its spatial dimensions. The gamma_initializer sets the initial weights of the layer
    #to a normla distribution with mean at 0 and standard deviation at 0.02.
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

  architecture based on Zhu et al., but also the original Image Transformation
  Network by Johnson'''
  def __init__(self):
    super(Generator, self).__init__()
    #structure as referenced by Zhu et al. (slight differences from Johnson architecture):
    #c7s1-64,d128,d256,R256,R256,R256, R256,R256,R256,u128,u64,c7s1-3

    #parameter for reflection padding before using a 7x7 kernel convolution
    self.padds = 3

    #Weights are initialized from a Gaussian distribution N(0, 0.02).
    kernel_initializer = tf.random_normal_initializer(stddev=0.02)

    #padd before using a 7x7 kernel as done in the first layer (Defined in strided block)
    self.padd1 = layers.ReflectionPadding2D(padding=(self.padds, self.padds))

    self.strided_block = [
      #Johnson: 32 x9 x9 conv, stride 1 -> 32??128??128
      #but in the cycle gan paper use 7x7, 64x128x128
      DownsampleBlock(nr_filters =64, kernel_size = 7, stride = 1, padding = 'valid', kernel_initializer = kernel_initializer),

      #Zhu: 128??3??3, conv, stride 2 -> 128??64??64
      DownsampleBlock(nr_filters =128, kernel_size = 3, stride = 2, padding = 'same',kernel_initializer = kernel_initializer),

      #Zhu: 256??3??3 conv, stride 2 -> 256??32??32
      DownsampleBlock(nr_filters =256, kernel_size = 3, stride = 2, padding = 'same', kernel_initializer = kernel_initializer)
    ]

    #Residual blocks -> 6 for 128x128 images (no padding in residual blocks: see text)
    self.blocks = [
      ResidualBlock(256, kernel_initializer = kernel_initializer),
      ResidualBlock(256, kernel_initializer = kernel_initializer),
      ResidualBlock(256, kernel_initializer = kernel_initializer),
      ResidualBlock(256, kernel_initializer = kernel_initializer),
      ResidualBlock(256, kernel_initializer = kernel_initializer),
      ResidualBlock(256, kernel_initializer = kernel_initializer) # 32x32
    ]

    self.transposed_block = [
      #128??3??3 conv, stride 1/2 -> 128??64??64
      UpsampleBlock(128,3,2, kernel_initializer),
      #64??3??3 conv, stride 1/2 -> 64??128??128
      UpsampleBlock(64,3,2, kernel_initializer),
    ]
    #padding: to keep size correct after next covolution
    self.padd2 = layers.ReflectionPadding2D(padding=(self.padds, self.padds))

    #3??7??7 conv, stride 1 -> 3??128??128
    self.final_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=7,
                                             strides =1, activation = tf.keras.activations.tanh,
                                             padding = 'valid', kernel_initializer = kernel_initializer)


  def call(self, x):
    x = self.padd1(x)

    for layer in self.strided_block:
      x = layer(x)

    for block in self.blocks:
      x = block(x)

    for layer in self.transposed_block:
      x = layer(x)


    x = self.padd2(x)

    x = self.final_layer(x)
    return x
