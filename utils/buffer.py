import numpy as np

'''Creates a buffer for storing generated images.

  Keyword Arguments:
  input: nr_images - how many images the buffer is supposed to store

  functions:
  get_image_buffer random elements of the current buffer (needed for training discriminator)
  set_image_buffer: adds images to the buffer (with a 50% chance at a random position if already full)'''
class Buffer():
  def __init__(self,nr_images):
    self.max_size = nr_images
    self.buffer_list = [0]*self.max_size
    self.next_index = 0

  def get_image_buffer(self):
    #return one of the images at random
    #random_id = np.random.randint(0, self.max_size - 1)
    #return [self.buffer_list[random_id]]
    return self.buffer_list

  #adds images to the buffer
  def set_image_buffer(self,generated_image):
  #saves generated images to the image_buffer
    for image in generated_image:
      #if self.buffer.length < self.max_size:
      #  buffer.append(image)
      #else:
       # p = random.random()
        #with a 50% probability store the image at a certain position
        #  if p > 0.5:
         #   random_id = random.randint(0, self.max_size - 1)
          #  self.buffer[random_id] = image
      if self.next_index >= self.max_size:
          self.next_index = 0

      self.buffer_list[self.next_index] = image
      self.next_index += 1
