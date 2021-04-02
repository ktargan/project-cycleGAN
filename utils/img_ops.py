import tensorflow as tf
import matplotlib.pyplot as plt
import io

def plot(image_batch, gen_images, gen_images_back):
    '''plots 3 images next to each other.

    Keyword arguments: original image,
                       generated image,
                       and backwards generated
    '''
    # Define plot with one ow and three columns
    fig, ax = plt.subplots(1,3)
    # Squeeze image will change the shape of the image from (1, hight, width, 3) to (hight, width, 3)
    # Convert image datatype to unit8, else mathplotlib does throw clipping error.
    # Multiply th eimage by *0.5+0.5 to undo the normalization (-1:1)
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
  '''Creates a plot each dataset it shows a full image cycle - i.e. our network translates
  an image from the domain of one domain X to domain Y and from domain Y back to X.

  Keyword arguments:
           dataset1, contains images of domain 1
	   dataset_2,  contains images of domain 2
           generator_1, translates images from domain 2 to domain 1 - i.e. generates domain 1 images
  	   generator_2, translates images from domain 1 to domain 2
  '''

  # For ablation studies we want to generate the same 30 test images for both datasets
  # So for ablation we set take_images to 30
  if ablation == True:
    take_images = 30
  else:
    # Inside the training process we plot one sample image cylcle for both datasets
    take_images = 1

  # Image generation cycle: from dataset 2 to dataset 1 and back
  for image_batch_2 in dataset2.take(take_images):
    # Feed original image into generator which translates it from domain 2 to domain 1
    gen_images_1 = generator_1(image_batch)
    # Feed generated image into generator which translates it from domain 1 to domain 2
    gen_images_back_2 = generator_2(gen_images_1)
    # Feed the resulting images into our plot function
    plot(image_batch_2, gen_images_1, gen_images_back_2)

  # Repeat the same process for dataset 1
  for image_batch in dataset1.take(take_images):
    gen_images = generator_2(image_batch)
    gen_images_back = generator_1(gen_images)
    plot(image_batch, gen_images, gen_images_back)


def plot_to_tf_image(generator_1, generator_2, dataset_1, dataset_2):
  '''Creates a 2x2 grid plot and converts it to one single tf image.

  Keyword arguments:
             dataset1, contains images of domain 1
	     dataset_2,  contains images of domain 2
             generator_1, translates images from domain 2 to domain 1 - i.e. generates domain 1 images
  	     generator_2, translates images from domain 1 to domain 2 

  Returns:   A png image converted to TF image
  '''  

  # Get one image for each datast
  for img_1, img_2 in tf.data.Dataset.zip((dataset_1, dataset_2)).take(1):
    
    # Generate image transformed from domain 1 to domain 2 and the other way around
    gen_image_1 = generator_1(img_2)
    gen_image_2 = generator_2(img_1)
    # Backward cycle
    gen_image_back_1 = generator_1(gen_image_2)
    gen_image_back_2 = generator_2(gen_image_1)

    # Create a figure that will contain our plot
    figure = plt.figure(figsize=(15,10))
    # Specifies the index on the grid
    ax = plt.subplot(2, 3, 1)
    # Squee to eliminate the batchsize dimension. 
    # Mutiply image by 0.5 and add 0.5 to undo the normalization
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(img_2*0.5 +0.5, dtype= tf.uint8)))
    plt.axis('off')
    ax = plt.subplot(2, 3, 2)
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(gen_image_1*0.5 +0.5, dtype= tf.uint8)))
    plt.axis('off')
    ax = plt.subplot(2, 3, 3)
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(gen_image_back_2*0.5 +0.5, dtype= tf.uint8)))
    plt.axis('off')
    ax = plt.subplot(2, 3, 4)
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(img_1*0.5 +0.5, dtype= tf.uint8)))
    plt.axis('off')
    ax = plt.subplot(2, 3, 5)
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(gen_image_2*0.5 +0.5, dtype= tf.uint8)))
    plt.axis('off')
    ax = plt.subplot(2, 3, 6)
    plt.imshow(tf.squeeze(tf.image.convert_image_dtype(gen_image_back_1*0.5 +0.5, dtype= tf.uint8)))
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



