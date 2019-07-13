import tensorflow as tf
import numpy as np
import csv

from .image import resize_image
from .visualization import draw_detections
from .. import models
from ..preprocessing.csv_generator import _read_classes

def csv_label_to_name_func(csv_classes_path):

    with open(csv_classes_path, 'r', newline='') as csv_file:
        classes = _read_classes(csv.reader(csv_file, delimiter=','))

    labels =  {}
    for key, value in classes.items():
        labels[value] = key

    def label_to_name(label):
        return labels[label]

    return label_to_name

def load_graph_def(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def

def generate_prediction_func(
        frozen_graph_filename,
        backbone_name,
        csv_classes_path,
        max_detections=100,
        score_threshold=0.05, # threshold score for showing prediction
    ):

    # graph_def = load_graph_def(frozen_graph_filename)

    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf_sess = tf.Session(config=tf_config)
    # tf.import_graph_def(graph_def, name='')

    preprocess_image = models.backbone(backbone_name).preprocess_image
    label_to_name = csv_label_to_name_func(csv_classes_path)

    # boxes_tensor = tf_sess.graph.get_tensor_by_name("ident_boxes/Identity:0")
    # scores_tensor = tf_sess.graph.get_tensor_by_name("ident_scores/Identity:0")
    # labels_tensor = tf_sess.graph.get_tensor_by_name("ident_labels/Identity:0")



    def pred_func(raw_image):
        image        = preprocess_image(raw_image.copy())
        image, scale = resize_image(image)
        image = np.expand_dims(image, axis=0)

        feed_dict = {
            "input_1:0": image
        }

        # boxes, scores, labels = tf_sess.run([boxes_tensor, scores_tensor, labels_tensor], feed_dict)
        # print("boxes: ", boxes.shape)
        # print("scores: ", scores.shape)
        # print("labels: ", labels.shape)

        # boxes /= scale
        # indices = np.where(scores[0, :] > score_threshold)[0]

        # scores = scores[0][indices]
        # scores_sort = np.argsort(-scores)[:max_detections]

        # image_boxes      = boxes[0, indices[scores_sort], :]
        # image_scores     = scores[scores_sort]
        # image_labels     = labels[0, indices[scores_sort]]

        ret_img = raw_image.copy()

        # draw_detections(
        #     ret_img,
        #     image_boxes,
        #     image_scores,
        #     image_labels,
        #     label_to_name=label_to_name)

        return ret_img

    return pred_func