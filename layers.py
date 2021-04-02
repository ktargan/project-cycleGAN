import tensorflow as tf

#code from keras cycle gan: (reference)
#Nain, A. K. (2020, August 12). Keras documentation: CycleGAN.
#Keras. https://keras.io/examples/generative/cyclegan/.
class ReflectionPadding2D(tf.keras.layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor."""

    def __init__(self, padding=(1, 1), **kwargs):
        #how much padding should be added (passed as argument)
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            #in prinicpal could add different padding amounts of padding in vertical
            #and horizontal direction
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        #finally call the pre-implemented tensorflow padding layer in reflection mode
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")
