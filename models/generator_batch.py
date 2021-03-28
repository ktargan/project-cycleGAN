import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import layers

'''Defines a Block as used in the ResNet architecture
  2 covolutional layers
  Batchnormalization
  Reflection Padding

  Keyword Arguments:
  nr_filters : amount of filters for the convolutional layers
  kernel_initializer: how to inititialize the weights
'''
class ResidualBlock(tf.keras.layers.Layer):
  def __init__(self, nr_filters, kernel_initializer):
    super(ResidualBlock,self).__init__()

    #use padding to keep the featuremap size constant even after applying convolutions
    self.padd1 = layers.ReflectionPadding2D()

    #3x3 conv, normalization, relu, 3x3 conv, normalization, relu
    self.conv_1 = tf.keras.layers.Conv2D(filters = nr_filters, kernel_size = 3,
                                         kernel_initializer = kernel_initializer
                                        #kernel_regularizer = tf.keras.regularizers.l2(0.01),
                                         #padding = 'same'
                                         )

    #instance normalization as batch size is 1
    self.batch_1 = tf.keras.layers.BatchNormalization()

    self.relu_1 = tf.keras.layers.ReLU()

    self.padd2 = layers.ReflectionPadding2D()
    self.conv_2 = tf.keras.layers.Conv2D(filters = nr_filters, kernel_size = 3,
                                         kernel_initializer = kernel_initializer
                                        # padding = 'same'
                                         )
    self.batch_2 = tf.keras.layers.BatchNormalization()

    self.relu_2 = tf.keras.layers.ReLU()

  def call(self, start_x, training):
    #x = tf.pad(start_x,[[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    x = self.padd1(start_x)
    x = self.conv_1(x)
    x = self.batch_1(x, training)

    x = self.relu_1(x)
    #x = tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    x = self.padd2(x)
    x = self.conv_2(x)
    x = self.batch_2(x, training)
    # skip connections are used: add up the input to block to its output
    x = x + start_x
    x = self.relu_2(x)
    return x


'''Block used for downsampling images/featuremaps with strided convolutions
  Convolutional layer, Batch Normalization, LeakyReLU activation

  Keyword Arguments:
  parameters needed to define convolutional layer'''
class DownsampleBlock(tf.keras.layers.Layer):
  def __init__(self, nr_filters, kernel_size, stride, padding, kernel_initializer):
    super(DownsampleBlock,self).__init__()

    self.conv = tf.keras.layers.Conv2D(nr_filters, kernel_size= kernel_size, strides = stride,
                                       padding=padding, kernel_initializer = kernel_initializer)
    self.norm_layer = tf.keras.layers.BatchNormalization()

    self.activation = tf.keras.layers.LeakyReLU()

  def call(self, x, training):

    x = self.conv(x)
    x = self.norm_layer(x, training)

    x = self.activation(x)

    return x


'''Block used for upsampling images with fractionally strided convolution,
  TransposedConvolutional layer, Batch Normalization, LeakyReLU activation

  Keyword Arguments:
  parameters needed to define convolutional layer'''
class UpsampleBlock(tf.keras.layers.Layer):
  def __init__(self, nr_filters, kernel_size, stride, kernel_initializer):
    super(UpsampleBlock,self).__init__()

    self.conv = tf.keras.layers.Conv2DTranspose(nr_filters, kernel_size= kernel_size, strides = stride,
                                       padding='same', kernel_initializer = kernel_initializer)
    self.norm_layer = tf.keras.layers.BatchNormalization()

    self.activation = tf.keras.layers.LeakyReLU()

  def call(self, x, training):

    x = self.conv(x)
    x = self.norm_layer(x, training)

    x = self.activation(x)

    return x



'''Generator is built up from different Down-, upsampling and Residual blocks

  architecture based on Zhu et al., but also the original Image Transformation
  Network by Johnson

  Keyword Arguments:
  '''
class Generator(tf.keras.Model):
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
      #Johnson: 32 x9 x9 conv, stride 1 -> 32×128×128
      #but in the cycle gan paper use 7x7, 64
      DownsampleBlock(nr_filters =64, kernel_size = 7, stride = 1, padding = 'valid', kernel_initializer = kernel_initializer),

      #Johnson: 64×3×3, conv, stride 2 -> 64×64×64
      #Zhu: 128x3x3
      DownsampleBlock(nr_filters =128, kernel_size = 3, stride = 2, padding = 'same',kernel_initializer = kernel_initializer),

      #Johnson: 128×3×3 conv, stride 2 -> 128×32×32
      #Zhu: 256
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
      #64×3×3 conv, stride 1/2 -> 64×64×64
      UpsampleBlock(128,3,2, kernel_initializer),
      #32×3×3 conv, stride 1/2 -> 32×128×128
      UpsampleBlock(64,3,2, kernel_initializer),
    ]
    #padding:
    self.padd2 = layers.ReflectionPadding2D(padding=(self.padds, self.padds))

    #3×9×9 conv, stride 1 3×128×128
    #Johnson et al. use scaled tanh (?) elsewhere use relu?
    self.final_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=7,
                                             strides =1, activation = tf.keras.activations.tanh,
                                             padding = 'valid', kernel_initializer = kernel_initializer)

    #self.concat = tf.keras.layers.Concatenate()

  def call(self, x, training):
    x = self.padd1(x)

    for layer in self.strided_block:
      x = layer(x, training)

    for block in self.blocks:
      x = block(x, training)

    for layer in self.transposed_block:
      x = layer(x, training)


    x = self.padd2(x)

    x = self.final_layer(x)
    return x
