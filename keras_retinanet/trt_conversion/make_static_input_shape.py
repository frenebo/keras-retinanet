import importlib
from keras import backend as K
import keras
import os
import sys
import argparse
import json
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
    parser = argparse.ArgumentParser(description='Make model have static input shape 1,900,900,3')
    parser.add_argument("source_model", help="Source model path")
    parser.add_argument("static_model_save", help="Path to save static model")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone name")

    args = parser.parse_args()

    old_model = models.load_model(args.source_model)
    config_dict = json.loads(old_model.to_json())
    config_dict["config"]["layers"][0]["config"]["batch_input_shape"] = [1, 900, 900, 3]

    new_model = keras.models.model_from_json(
        json.dumps(config_dict),
        custom_objects=models.backbone(args.backbone).custom_objects,
    )

    new_model.set_weights(old_model.get_weights())

    new_model.save(args.static_model_save)

if __name__ == "__main__":
    main()
