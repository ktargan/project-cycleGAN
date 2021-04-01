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


def plot_to_tf_image(generator_1, generator_2, dataset_1, dataset_2):
  """Creates a 2x2 grid plot and converts it to one singe PNG image.
  Arguments: dataset1, contains images of domain 1
	     dataset_2,  contains images of domain 2
             generator_1, translates images from domain 2 to domain 1 
	       - i.e. generates domain 1 images
  	     generator_2, translates images from domain 1 to domain 2 
  Returns:   A png image converted to TF image"""  

  # Get one image for each datast
  for img_1, img_2 in tf.data.Dataset.zip((dataset_1, dataset_2)).take(1):
    
    # Generate image transformed from domain 1 to domain 2 and the other way around
    gen_image_1 = generator_1(img_2)
    gen_image_2 = generator_2(img_1)

    # Create a figure that will contain our plot
    figure = plt.figure(figsize=(20,20))
    # Specifies the index on the grid
    ax = plt.subplot(2, 2, 1)
    # Squee to eliminate the batchsize dimension. 
    # Mutiply image by 0.5 and add 0.5 to undo the normalization
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(img_2*0.5 +0.5, dtype= tf.uint8)))
    plt.axis('off')
    ax = plt.subplot(2, 2, 2)
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(gen_image_1*0.5 +0.5, dtype= tf.uint8)))
    plt.axis('off')
    ax = plt.subplot(2, 2, 3)
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(img_1*0.5 +0.5, dtype= tf.uint8)))
    plt.axis('off')
    ax = plt.subplot(2, 2, 4)
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(gen_image_2*0.5 +0.5, dtype= tf.uint8)))
    plt.axis('off')

    # We copied the section below for converting the figure to PNG 
    # from the tensorboard documentarion at:
    # https://www.tensorflow.org/tensorboard/image_summaries#visualizing_multiple_images

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image with 3 channels for RGB
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image



