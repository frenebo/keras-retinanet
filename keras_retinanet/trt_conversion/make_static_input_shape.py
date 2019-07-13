import importlib
from keras import backend as K
import keras
import os
import sys
import argparse
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

import tensorflow as tf


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.trt_conversion  # noqa: F401
    __package__ = "keras_retinanet.trt_conversion"

from .. import models

def main():
    parser = argparse.ArgumentParser(description='Make model have static input shape 1,224,224,3')
    parser.add_argument("source_model", help="Source model path")
    parser.add_argument("static_model_save", help="Path to save static model")

    args = parser.parse_args()

    model = models.load_model(args.source_model)

    new_input = keras.layers.Input(batch_shape=(1, 224, 224, 3))

    new_model = keras.models.clone_model(model, input_tensors=new_input)

    # model.layers[0] = new_input

    new_model.save(args.static_model_save)

if __name__ == "__main__":
    main()
