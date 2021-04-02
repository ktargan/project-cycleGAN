import tensorflow as tf
import tensorflow_datasets as tfds


def get_oranges(batchsize):
    '''Function loads a oranges from apple2orange dataset and starts preprocessing

        Keyword arguments:
        batchsize

        returns preprocessed datasets'''
    #loads the orange dataset used for CycleGAN apple2orange
    train_oranges, test_oranges = tfds.load('cycle_gan/apple2orange', split = ['trainB', 'testB[:30]'], as_supervised=True)

    #pass dataset through pipeline
    train_oranges = preprocessing(train_oranges, batchsize, do_variation = True)
    test_oranges = preprocessing(test_oranges, batchsize, do_variation = False)
    return train_oranges, test_oranges


def get_horses(batchsize):
    '''Function load a horse2zebra dataset and starts preprocessing

        Keyword arguments:
        batchsize

        returns: preprocessed datasets'''

    train_horses, train_zebras, test_horses, test_zebras = tfds.load('cycle_gan/horse2zebra',
                                                                 split = ['trainA','trainB', 'testA[:30]', 'testB[:30]'],
                                                                 as_supervised=True)

    #perform further preprocessing steps
    train_horses = preprocessing(train_horses, batchsize, do_variation = True)
    train_zebras = preprocessing(train_zebras, batchsize,  do_variation = True)

    #for the test dataset which we use to print images in the end
    test_horses = preprocessing(test_horses, batchsize,  do_variation = False)
    test_zebras = preprocessing(test_zebras, batchsize,  do_variation = False)
    return train_horses, train_zebras, test_horses, test_zebras


def get_custom(path,batchsize, copy_times):
    '''Function will load a custom dataset and preprocess it.

        Keyword arguments:
        path to load from
        batchsize
        copy_times : how often shall the small dataset be copied and augmented to create larger datset

        returns: preprocessed dataset'''

    fantasy_dataset = tf.keras.preprocessing.image_dataset_from_directory(path, image_size= (220,220),
                                label_mode= None, shuffle = False, batch_size =batchsize)


    #For us these training sets were small (12-24 images)
    #thus we filled up our datasets with either exact copies or crops of the images, so:
    #Copy images in style refernce / fantasy dataset: and randomly crop some of the copies
    fantasy_dataset_1 = fantasy_dataset.map(lambda image: tf.image.resize(image,[128,128]))
    for i in range(int(round(copy_times/2))):
      fantasy_dataset_1 = fantasy_dataset_1.concatenate(fantasy_dataset.map(lambda image: tf.image.resize(image,[128,128])))

    for i in range(int(round(copy_times/2))):
      fantasy_dataset_1 = fantasy_dataset_1.concatenate(fantasy_dataset.map(lambda image: tf.image.random_crop(image,[1,128,128,3])))

    fantasy_dataset = fantasy_dataset_1.map(lambda image: tf.image.random_flip_left_right(image))

    # images are normalizied to [-1, 1]
    fantasy_dataset = fantasy_dataset.map(lambda image: (image/127.5)-1)

    #Zhu et al. use a batchsize of 1
    fantasy_dataset = fantasy_dataset.shuffle(buffer_size = 1000)
    fantasy_dataset = fantasy_dataset.prefetch(8)

    return fantasy_dataset



def preprocessing(image_set, batchsize, do_variation):
    '''Function defines input pipeline for preprocessing

    Keyword arguments:
    batchsize
    do_variation: boolean that indicates if the dataset be slightly variated

    returns: preprocessed dataset'''
    #if images should be preprocessed with slightly random variations
    if do_variation:
        #resize image to smaller size (faster computation and thus more manageable for the scope of the task)
        #firstly by simply resizing and secondly randomly cropping the resulting images (introduces variation)
        image_set = image_set.map(lambda image, label: tf.image.resize(image,[135,135]))
        image_set = image_set.map(lambda image: tf.image.random_crop(image,[128,128,3]))

        #randomly decide to mirror images (make sure that they do not all face the same direction for one class)
        image_set = image_set.map(lambda image: tf.image.random_flip_left_right(image))

        # images are normalizied to [-1, 1]
        image_set = image_set.map(lambda image: (image/127.5)-1)

        image_set = image_set.shuffle(buffer_size = 1000)
    #in case of test dataset skip variation step
    else:
        image_set = image_set.map(lambda image, label: tf.image.resize(image,[128,128]))
        image_set = image_set.map(lambda image: (image/127.5)-1)
    #Zhu et al. use a batchsize of 1
    image_set = image_set.batch(batchsize)
    image_set = image_set.prefetch(8)

    return image_set
