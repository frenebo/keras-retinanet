import tensorflow as tf
import numpy as np
import csv
import cv2

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

    print("Loading graph definition... ", end="")
    graph_def = load_graph_def(frozen_graph_filename)
    print("Done loading graph definition")

    print("Creating TF session... ", end="")
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    print("Done creating TF session")

    print("Importing graph definition... ", end="")
    tf.import_graph_def(graph_def, name='')
    print("Done importing graph_definition ", end="")

    preprocess_image = models.backbone(backbone_name).preprocess_image
    label_to_name = csv_label_to_name_func(csv_classes_path)

    print("Getting graph output tensors... ", end="")
    boxes_tensor = tf_sess.graph.get_tensor_by_name("ident_boxes/Identity:0")
    scores_tensor = tf_sess.graph.get_tensor_by_name("ident_scores/Identity:0")
    labels_tensor = tf_sess.graph.get_tensor_by_name("ident_labels/Identity:0")
    print("Done getting graph output tensors")


    def pred_func(raw_image):
        print("Preprocessing image... ", end="")
        image        = preprocess_image(raw_image.copy())
        print("Image shape: ", image.shape, end=" ")
        x_scale = 224 / image.shape[1]
        y_scale = 224 / image.shape[0]

        image = cv2.resize(image, None, fx=x_scale, fy=y_scale)

        # image, scale = resize_image(image, min_side=200, max_side=300)
        image = np.expand_dims(image, axis=0)
        print("Done preprocessing image")

        feed_dict = {
            "input_1:0": image
        }

        print("Running TF session on image... ", end="", flush=True)
        boxes, scores, labels = tf_sess.run([boxes_tensor, scores_tensor, labels_tensor], feed_dict)
        print("Done running tf session on image")
        print("boxes: ", boxes.shape)
        # print("scores: ", scores.shape)
        # print("labels: ", labels.shape)

        print("Extracting predictions from session output... ", end="")
        print("Boxes shape: ", boxes.shape)
        # boxes /= scale
        indices = np.where(scores[0, :] > score_threshold)[0]

        scores = scores[0][indices]
        scores_sort = np.argsort(-scores)[:max_detections]

        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        print("Done extracting predictions")

        print("Drawing detections... ", end="")
        ret_img = raw_image.copy()
        draw_detections(
            ret_img,
            image_boxes,
            image_scores,
            image_labels,
            label_to_name=label_to_name)
        print("Done drawing detections")

        return ret_img

    return pred_func