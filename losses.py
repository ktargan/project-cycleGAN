import tensorflow as tf
# for neural style transfer: "try to capture correspondences between higher-level appearance structures"

#cycle consistency loss: used as a way to introduce a sort of supervision
#both Generators should work consistently
def calc_cycle_loss(real_image, cycled_image):
  #summed up distances between generated images and real starting image
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return 1 * loss1

#adversarial loss
# match distribution of generated images to target (ground truth) distribution
#LGAN(G,DY,X, Y) = Ey∼pdata(y)[logDY(y)] + Ex∼pdata(x)[log(1 −DY(G(x))]
# but use least squares loss instead of negative log likelihood
def generator_loss(prediction):
  return tf.reduce_mean(tf.squared_difference(prediction, 1))

def discriminator_loss(generated_image, real_image):
  #real_img_labels: should have been predicted as ones
  #fake as 0s
  #least squares loss
  real_loss = tf.reduce_mean(tf.squared_difference(real_image, 1))
  fake_loss = tf.reduce_mean(tf.squared_difference(generated_image, 0))

  return (real_loss + fake_loss)*0.5
