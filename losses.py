import tensorflow as tf
BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# for neural style transfer: "try to capture correspondences between higher-level appearance structures"

#cycle consistency loss: used as a way to introduce a sort of supervision
#both Generators should work consistently
def calc_cycle_loss(real_image, cycled_image, lamba):
  #summed up distances between generated images and real starting image
  loss1 = tf.math.reduce_mean(tf.math.abs(real_image - cycled_image))

  return (lamba * loss1)

#adversarial loss
# match distribution of generated images to target (ground truth) distribution
# LGAN(G,DY,X, Y) = Ey∼pdata(y)[logDY(y)] + Ex∼pdata(x)[log(1 −DY(G(x))]
# but use least squares loss instead of negative log likelihood
def generator_loss(prediction):
  gen_loss =  tf.math.reduce_mean(tf.math.squared_difference(prediction, 1))
  #gen_loss =  binary_cross_loss(tf.ones_like(prediction), prediction)
  return gen_loss

def discriminator_loss(generated_image, real_image):
  #real_img_labels: should have been predicted as ones
  #fake as 0s
  #least squares loss
  real_loss = tf.math.reduce_mean(tf.math.squared_difference(real_image, 1))
  fake_loss = tf.math.reduce_mean(tf.math.squared_difference(generated_image, 0))
  #real_loss = binary_cross_loss(tf.ones_like(real_image), real_image)
  #fake_loss = binary_cross_loss(tf.zeros_like(generated_image), generated_image)

  return (real_loss + fake_loss)*0.5

# check if the generator of domain_A keeps the image of domain_A similar
def identity_loss(real_image, same_image, lamba):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return (lamba* 0.5 * loss)


def bce_gen_loss(prediction):
  gen_loss =  BCE(tf.ones_like(prediction), prediction)
  return gen_loss

def bce_discrim_loss(generated_image, real_image):
  #real_img_labels: should have been predicted as ones
  #fake as 0s
  #binary corss entropy loss
  real_loss = BCE(tf.ones_like(real_image), real_image)
  fake_loss = BCE(tf.zeros_like(generated_image), generated_image)
  return (real_loss + fake_loss)*0.5
