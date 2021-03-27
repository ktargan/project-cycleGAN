import tensorflow as tf
import matplotlib.pyplot as plt

def plot(image_batch, gen_images, gen_images_back, title):
  #plots 3 images next to each other, original, generated and backwards generated
    fig, ax = plt.subplots(1,3)
    # convert image datatype, so that mathplotlib does not throw clipping error
    #fig.subtitle(title, fontsize=9)
    ax[0].imshow(tf.squeeze(tf.image.convert_image_dtype(image_batch[0]*0.5 +0.5, dtype= tf.uint8)))
    ax[0].set_title("original image")
    ax[0].axis("off")
    ax[1].imshow(tf.squeeze(tf.image.convert_image_dtype(gen_images[0]*0.5 +0.5, dtype= tf.uint8)))
    ax[1].set_title("generated")
    ax[1].axis("off")
    ax[2].imshow(tf.squeeze(tf.image.convert_image_dtype(gen_images_back[0]*0.5 +0.5, dtype= tf.uint8)))
    ax[2].set_title("backwards generated")
    ax[2].axis("off")
    plt.show()

def plot_image_cycle(generator_1, generator_2, dataset1, dataset2, ablation = False):
  # for ablation studies we want to generate the same 30 test images for both classes
  # thus we need to take 30 images from the dataset
  # in the training process we only want to see one sample image cylcle from both classes
  if ablation == True:
    take_images = 30
  else:
    take_images = 1

  # image cycle dataset1 to 2 (and back)
  for image_batch in dataset2.take(take_images):
    gen_images = generator_1(image_batch)
    gen_images_back = generator_2(gen_images)
    #plots the images with description labels
    plot(image_batch, gen_images, gen_images_back, title = "Generateor 1 cycle: ")

  # image cycle dataset2 to 1 (and back)
  for image_batch in dataset1.take(take_images):
    gen_images = generator_2(image_batch)
    gen_images_back = generator_1(gen_images)
    plot(image_batch, gen_images, gen_images_back, title = "Generator 2 cycle: ")
