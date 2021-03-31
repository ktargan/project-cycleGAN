import tensorflow as tf
import tensorflow_datasets as tfds

def get_oranges():
    train_oranges = tfds.load('cycle_gan/apple2orange', split = ['trainB'],
                                                                 as_supervised=True)
    return train_oranges

def get_fantasy(path):
    fantasy_dataset = tf.keras.preprocessing.image_dataset_from_directory(path, image_size= (220,220),
                                label_mode= None, shuffle = False, batch_size = 1)
    #For us these training sets were small (12-24 images)
    #thus we filled up our datasets with either exact copies or crops of the images, so:
    #Copy images in style refernce / fantasy dataset: and randomly crop some of the copies
    fantasy_dataset_1 = fantasy_dataset.map(lambda image: tf.image.resize(image,[128,128]))
    for i in range(30):
      fantasy_dataset_1 = fantasy_dataset_1.concatenate(fantasy_dataset.map(lambda image: tf.image.resize(image,[128,128])))

    for i in range(40):
      fantasy_dataset_1 = fantasy_dataset_1.concatenate(fantasy_dataset.map(lambda image: tf.image.random_crop(image,[1,128,128,3])))

      return fantasy_dataset_1
