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
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, ZeroPadding2D

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

    x = inputs

    x = ZeroPadding2D(padding=(3, 3), name='first_zeropad')(x)
    x = Conv2D(64, (7, 7),
                      strides=(3, 3),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='first_conv')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu', name="first_activation")(x)
    # x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='first_maxpooling')(x)

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

def _resnet_identity_block(
    input_tensor,
    kernel_size,
    filters,
    stage,
    block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Layers
    x = keras.layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = keras.layers.Add(name="identity_add_stage_{}_block_{}".format(stage, block))([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def _resnet_conv_block(
    input_tensor,
    kernel_size,
    filters,
    stage,
    block,
    strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """

    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Layers
    x = input_tensor

    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = keras.layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = keras.layers.Add(name="resnet_conv_block_stage_{}_block_{}".format(stage, block))([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x

def create_resnet_and_return_outputs(num_classes, backbone='other', inputs=None, modifier=None, **kwargs):
    if not trainable:
        raise Exception("Unimplemented non-trainable resnet")
    bn_axis = 3

    # Layers

    x = inputs
    x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = keras.layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Convolution module 2
    x = _resnet_conv_block(x, 3, filters=[64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = _resnet_identity_block(x, 3, filters=[64, 64, 256], stage=2, block='b')
    C2_output = x = _resnet_identity_block(x, 3, filters=[64, 64, 256], stage=2, block='c')

    # Convolution module 3
    x = _resnet_conv_block(x, 3, filters=[128, 128, 512], stage=3, block='a')
    x = _resnet_identity_block(x, 3, filters=[128, 128, 512], stage=3, block='b')
    x = _resnet_identity_block(x, 3, filters=[128, 128, 512], stage=3, block='c')
    C3_output = x = _resnet_identity_block(x, 3, filters=[128, 128, 512], stage=3, block='d')

    # Convolution module 4
    x = _resnet_conv_block(x, 3, filters=[256, 256, 1024], stage=4, block='a')
    x = _resnet_identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='b')
    x = _resnet_identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='c')
    x = _resnet_identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='d')
    x = _resnet_identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='e')
    C4_output = x = _resnet_identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='f')

    # Convolution module 5
    x = _resnet_conv_block(x, 3, filters=[512, 512, 2048], stage=5, block='a')
    x = _resnet_identity_block(x, 3, filters=[512, 512, 2048], stage=5, block='b')
    C5_output = x = _resnet_identity_block(x, 3, filters=[512, 512, 2048], stage=5, block='c')

    model = Model(inputs=inputs, outputs=C5_output)

    if modifier:
        model = modifier(model)

    # output_layers =
    layer_outputs = [
        C3_output,
        C4_output,
        C5_output,
    ]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)


    # return {
    #     # "C2_output": C2_output, # Is this used?
    #     "C3_output": C3_output,
    #     "C4_output": C4_output,
    #     "C5_output": C5_output,
    # }