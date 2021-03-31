import tensorflow as tf
BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#cycle consistency loss: used to introduce a sort of supervision
#both Generators should work consistently
def calc_cycle_loss(real_image, cycled_image, lamba):
  #summed up distances between generated images and real starting image
  loss1 = tf.math.reduce_mean(tf.math.abs(real_image - cycled_image))
  #loss is weighted with lambda
  return (lamba * loss1)

#adversarial losses
# but use least squares loss instead of negative log likelihood
def generator_loss(prediction):
# generator aims to fool discriminator, thus the ideal label assignment
# from the perspective of the generator is "1"
  gen_loss =  tf.math.reduce_mean(tf.math.squared_difference(prediction, 1))
  return gen_loss

def discriminator_loss(generated_image, real_image):
  #real_img_labels: should have been predicted as ones
  #fake labels: should be predicted as 0s
  #least squares loss
  real_loss = tf.math.reduce_mean(tf.math.squared_difference(real_image, 1))
  fake_loss = tf.math.reduce_mean(tf.math.squared_difference(generated_image, 0))

  return (real_loss + fake_loss)*0.5

# check how the generator of domain_A (e.g. zebra) transforms an image from its target domain_A (zebra)
# the output (e.g. zebra) image should be similar underlying (e.g. zebra) image
def identity_loss(real_image, same_image, lamba):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return (lamba* 0.5 * loss)

# adversarial losses with negative log likelihood
def bce_gen_loss(prediction):
  gen_loss =  BCE(tf.ones_like(prediction), prediction)
  return gen_loss

def bce_discrim_loss(generated_image, real_image):
  real_loss = BCE(tf.ones_like(real_image), real_image)
  fake_loss = BCE(tf.zeros_like(generated_image), generated_image)
  return (real_loss + fake_loss)*0.5
