import numpy as np
import keras

class Argmax(keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return keras.backend.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:self.axis] + input_shape[self.axis+1:]
        return tuple(output_shape)

    def get_config(self):
        config = super(Argmax, self).get_config()
        config.update({
            "axis": self.axis
        })

        return config
