import tensorflow as tf
BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def calc_cycle_loss(real_image, cycled_image, lamba):
    ''' Defines the cycle consistency loss calculation, which ensures that generated
    images are based on input images (Generators work consistently)

    Keyword arguments:
    real_image: the underlying image that was subsequently passed into the first generators
        to produce generated image 1
    cycled_image: image that was generated by second generator (on the basis of the
        image 1 generated by first generator)
    lambda: weighting factor which influences the impact of this objective

    returns: calculated loss'''

    #summed up distances between generated images and real starting image
    loss1 = tf.math.reduce_mean(tf.math.abs(real_image - cycled_image))
    #loss is weighted with lambda
    return (lamba * loss1)

#adversarial losses
def generator_loss(prediction):
    ''' Defines the adversarial loss objective for the generator

    Keyword arguments:
    prediction: how the generated image was classified by the discriminator

    returns: calculated loss'''
    # generator aims to fool discriminator, thus the ideal label assignment
    # from the perspective of the generator is "1"
    # uses least squares loss instead of negative log likelihood
    gen_loss =  tf.math.reduce_mean(tf.math.squared_difference(prediction, 1))
    return gen_loss

def discriminator_loss(generated_image, real_image):
    ''' Defines the adversarial loss objective for the discriminator

    Keyword arguments:
    generated images: how the discriminator classified images generated by the respective Generators
     (should be '0's)
    real image: discriminators classification of real images (should be '1's)

    returns: calculated loss
    '''

    #least squares loss
    real_loss = tf.math.reduce_mean(tf.math.squared_difference(real_image, 1))
    fake_loss = tf.math.reduce_mean(tf.math.squared_difference(generated_image, 0))

    return (real_loss + fake_loss)*0.5

# check how the generator of domain_A (e.g. zebra) transforms an image from its target domain_A (zebra)
# the output (e.g. zebra) image should be similar underlying (e.g. zebra) image
def identity_loss(real_image, same_image, lamba):
    ''' Defines the identity loss calculation, which is sometimes added to further
    incentivize close preservation of input features

    Keyword arguments:
    real_image: the underlying image that was subsequently passed into the first generators
        to produce generated image 1
    same_image: image that was generated by first generator : is of same domain as real_image
    lambda: weighting factor which influences the impact of this objective

    returns: calculated loss
    '''
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return (lamba* 0.5 * loss)

# adversarial losses with negative log likelihood
def bce_gen_loss(prediction):
    ''' Defines the adversarial loss objective as negative log likelihood equation for the generator

    Keyword arguments:
    prediction: how the generated image was classified by the discriminator

    returns: calculated loss
    '''
    gen_loss =  BCE(tf.ones_like(prediction), prediction)
    return gen_loss

def bce_discrim_loss(generated_image, real_image):
    ''' Defines the adversarial loss objective as negative log likelihood equation for the discriminator

    Keyword arguments:
    generated images: how the discriminator classified images generated by the respective Generators
     (should be '0's)
    real image: discriminators classification of real images (should be '1's)

    returns: calculated loss
    '''
    real_loss = BCE(tf.ones_like(real_image), real_image)
    fake_loss = BCE(tf.zeros_like(generated_image), generated_image)
    return (real_loss + fake_loss)*0.5
