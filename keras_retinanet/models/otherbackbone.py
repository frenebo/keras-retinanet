"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import keras
from keras.utils import get_file
from keras.models import Model
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image

class OtherBackbone(Backbone):
    """ Describes backbone information and provides utility funcitons.
    """

    def retinanet(self, *args, **kwargs):
        return otherbackbone_retinanet(*args, backbone=self.backbone, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ["other"]

        if self.backbone not in allowed_backbones:
            raise ValueError("Backbone (\'{}\') not in allowed backbones ({})".format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        return preprocess_image(inputs, mode="caffe")

def otherbackbone_retinanet(num_classes, backbone='other', inputs=None, modifier=None, **kwargs):
    if backbone != "other":
        raise ValueError("Backbone {} not recognized.".format(backbone))

    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = Conv2D(64, (7, 7),
                      strides=(3, 3),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='first_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    # x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = inputs
    for i in range(4):
        x = Conv2D(64, (1, 1),
                        kernel_initializer='he_normal',
                        name="conv_{}".format(i))(x)
        x = BatchNormalization(axis=3, name="bn_{}".format(i))(x)
        x = Activation('relu', name="activation_{}".format(i))(x)

    model = Model(inputs=inputs, outputs=x)


    if modifier:
        model = modifier(model)

    layer_names = [
        "conv_1",
        "conv_2",
        "conv_3",
    ]
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
