# Importing TensorRT makes it possible to load TRT engine operations from saved graph
import tensorflow.contrib.tensorrt as trt

import tensorflow as tf
import numpy as np
import csv
import cv2
import datetime

from .image import resize_image
from .visualization import draw_detections
from .. import models
from ..preprocessing.csv_generator import _read_classes
from ..utils import SCALED_SIZE

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
        keep_downsized=False,
        limit_threads=None, # Either none or a number of threads
    ):

    print("Loading graph definition... ", end="")
    graph_def = load_graph_def(frozen_graph_filename)
    print("Done loading graph definition")

    print("Creating TF session... ", end="")
    if limit_threads is None:
        thread_args = {}
    else:
        thread_args = {
            "intra_op_parallelism_threads": limit_threads,
            "inter_op_parallelism_threads": limit_threads,
        }
    tf_config = tf.ConfigProto(
        log_device_placement=True,
        **thread_args,
    )
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    print("Done creating TF session")

    print("Importing graph definition... ", end="")
    tf.import_graph_def(graph_def, name='')
    print("Done importing graph_definition ", end="")

    # preprocess_image = models.backbone(backbone_name).preprocess_image
    label_to_name = csv_label_to_name_func(csv_classes_path)

    print("Getting graph output tensors... ", end="")
    boxes_tensor = tf_sess.graph.get_tensor_by_name("ident_boxes/Identity:0")
    scores_tensor = tf_sess.graph.get_tensor_by_name("ident_scores/Identity:0")
    labels_tensor = tf_sess.graph.get_tensor_by_name("ident_labels/Identity:0")
    print("Done getting graph output tensors")

    # model = models.load_model(model_path, backbone_name=backbone_name)
    # bbox_model = models.convert_model(model)


    def pred_func(raw_image):
        print("Preprocessing image... ", end="")
        image        = preprocess_image(raw_image.copy())
        print("Image shape: ", image.shape, end=" ")
        x_scale = SCALED_SIZE / image.shape[1]
        y_scale = SCALED_SIZE / image.shape[0]

        image = cv2.resize(image, None, fx=x_scale, fy=y_scale)

        # image, scale = resize_image(image, min_side=200, max_side=300)

        print("Done preprocessing image")

        feed_dict = {
            "input_1:0": np.expand_dims(image, axis=0)
        }

        print("Running model on image... ", end="", flush=True)
        start = datetime.datetime.now()
        # boxes, scores, labels = bbox_model.predict(np.expand_dims(image, axis=0))
        # boxes = boxes[0]
        # scores = scores[0]
        # labels = labels[0]
        boxes, scores, labels = tf_sess.run([boxes_tensor, scores_tensor, labels_tensor], feed_dict)
        end = datetime.datetime.now()
        milliseconds = (end - start).total_seconds()*1000
        print("Done running model on image, took {} milliseconds".format(milliseconds))

        # boxe values are ordered: x1, y1, x2, y2

        if not keep_downsized:
            boxes[:,:,0] /= x_scale
            boxes[:,:,2] /= x_scale
            boxes[:,:,1] /= y_scale
            boxes[:,:,3] /= y_scale

        print("Extracting predictions from session output... ")
        print("Boxes: ", boxes.shape)
        print("Scores: ", scores.shape)
        print("Labels: ", labels.shape)

        # boxes /= scale
        indices = np.where(scores[0, :] > score_threshold)[0]
        print("Indices: ", indices.shape)

        scores = scores[0][indices]
        scores_sort = np.argsort(-scores)[:max_detections]
        print("Scores: ", scores.shape)
        print("Indices[scores_sort]", indices[scores_sort].shape)

        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        print("Done extracting predictions")

        print("Drawing detections... ", end="")
        if keep_downsized:
            ret_img = image.copy()
        else:
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