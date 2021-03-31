import tensorflow as tf

#module defines training step for generator and discriminator
#variable names are defined in relation to the horse2zebra dataset to be more understandable

def training_step_discrim(discriminator, optimizer, images, generated_images):
  # calculate the discriminator loss and apply gradients
  with tf.GradientTape() as tape:
    # feed real images into discriminator, get the predictions
    real_image_predictions = discriminator(images)

    # feed fake images into discriminator, get the predictions
    fake_image_predictions = discriminator(generated_images)

    #calculate adversarial loss
    discr_loss = losses.discriminator_loss(fake_image_predictions, real_image_predictions)

    gradients = tape.gradient(discr_loss, discriminator.trainable_variables)

  optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
  return discr_loss

@tf.function
def training_step_gen(generator_zebras, generator_horses, discriminator_zebras, discriminator_horses,
                      images_zebras, images_horses, optimizer_zebras, optimizer_horses):
  #clarification: generator_zebras generates zebra images from horses
  #Calculate the loss for both generators and update the weights
  with tf.GradientTape() as tape_horse, tf.GradientTape() as tape_zebra:

    #feed original images to generators
    fake_images_zebras = generator_zebras(images_horses)
    fake_images_horses = generator_horses(images_pattern)

    #get the assigned predicition from the discriminators
    fake_image_predictions_zebras = discriminator_zebras(fake_images_zebras)
    fake_image_predictions_horses = discriminator_horses(fake_images_horses)

    #calculate the adversarial generatorloss:
    #did the discriminator recognize the images as generated?
    gen_loss_zebras = losses.generator_loss(fake_image_predictions_zebras)
    gen_loss_horses = losses.generator_loss(fake_image_predictions_horses)

    #pass the generetaed zebra images of generator_zebras to generator_horses
    #(to see if it produces horse images close to the original image)
    recreated_images_horses = generator_horses(fake_images_zebras)
    recreated_images_zebras = generator_zebras(fake_images_horses)

    #calculate cycle loss: the weighting factor lambda is set to 10
    #how much does the original image differ from the the cycled image
    cycle_loss_forward = losses.calc_cycle_loss(images_zebras, recreated_images_zebras, 15)
    cycle_loss_backward = losses.calc_cycle_loss(images_horses, recreated_images_horses, 15)
    total_cycle_loss = cycle_loss_forward + cycle_loss_backward

    #give images from their target domain to the generators
    # e.g. give zebra images to a zebra generator and then see if the output
    #images are close to original images -> identity loss
    same_images_reconstructed_zebras = generator_zebras(images_zebras)
    same_images_reconstructed_horses = generator_horses(images_horses)

    identity_loss_horses = losses.identity_loss(images_horses, same_images_reconstructed_horses, 15)
    identity_loss_zebras = losses.identity_loss(images_zebras, same_images_reconstructed_zebras, 15)

    # sum up the losses for each generator
    # this means the respective generator and identity loss (for their domain)
    # but also the complete cycle consistency loss!
    total_loss_zebras = gen_loss_zebras + total_cycle_loss + identity_loss_zebras
    total_loss_horses = gen_loss_horses + total_cycle_loss + identity_loss_horses

    #update weights (by calculating gradients) of the currently trained generator
    gradients_zebras = tape_zebra.gradient(total_loss_zebras, generator_zebras.trainable_variables)
    gradients_horses = tape_horse.gradient(total_loss_horses, generator_horses.trainable_variables)

  #update weights
  optimizer_zebras.apply_gradients(zip(gradients_zebras, generator_zebras.trainable_variables))
  optimizer_horses.apply_gradients(zip(gradients_horses, generator_horses.trainable_variables))

  #return loss and generated images for the buffer
  return total_loss_zebras, total_loss_horses, fake_images_zebras, fake_images_horses
