
from .generator import Generator
from ..utils.image import read_image_bgr
from .csv_generator import _open_for_csv, _read_classes

import xml.etree.ElementTree

import numpy as np
from PIL import Image

import csv
import os


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def find_bounding_box_of_points(
    points # list of Point classes
    ):
    min_x = None
    min_y = None
    max_x = None
    max_y = None

    for pt in points:
        if min_x is None or pt.x < min_x:
            min_x = pt.x
        if max_x is None or pt.x > max_x:
            max_x = pt.x

        if min_y is None or pt.y < min_y:
            min_y = pt.y
        if max_y is None or pt.y > max_y:
            max_y = pt.y

    return [min_x, min_y, max_x, max_y]

def _process_annotations(xml_text):
    annotation_el = xml.etree.ElementTree.fromstring(xml_text)
    filename_string = annotation_el.find("filename").text

    annotations = []

    for object_el in annotation_el.findall("object"):
        obj_name = object_el.find("name").text

        points = []
        for point_el in object_el.find("polygon").findall("pt"):
            x = float(point_el.find("x").text)
            y = float(point_el.find("y").text)

            point = Point(x, y)

            points.append(point)

        bounding_box = find_bounding_box_of_points(points)

        direction = object_el.find("attributes").text

        annotation = {
            "name": obj_name,
            "bounding_box": bounding_box,
            "direction": direction,
        }

        annotations.append(annotation)

    return {
        "filename": filename_string,
        "annotations": annotations
    }


def _load_images_annotations(annotations_dir):
    file_names = [f for f in os.listdir(annotations_dir) if os.path.isfile(os.path.join(annotations_dir, f))]

    annotations = []

    for file_name in file_names:
        with open(os.path.join(annotations_dir, file_name), "r") as xml_file:
            annotation = _process_annotations(xml_file.read())
        annotations.append(annotation)

    return annotations


from ..utils.anchors import compute_gt_annotations, anchor_targets_bbox, bbox_transform
import keras

def north_south_anchor_targets_bbox(
    anchors,
    image_group,
    annotations_group,
    num_classes,
    num_directions=2,
    negative_overlap=0.4,
    positive_overlap=0.5
):
    assert(len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."
        assert('directions' in annotations), "Annotations should contain north-south directions."

    batch_size = len(image_group)

    regression_batch  = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=keras.backend.floatx())
    labels_batch      = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())
    directions_batch  = np.zeros((batch_size, anchors.shape[0], num_directions + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_batch[index, ignore_indices, -1]       = -1
            labels_batch[index, positive_indices, -1]     = 1

            regression_batch[index, ignore_indices, -1]   = -1
            regression_batch[index, positive_indices, -1] = 1


            directions_batch[index, ignore_indices, -1]       = -1
            directions_batch[index, positive_indices, -1]     = 1

            # compute target class labels
            labels_batch[    index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1
            directions_batch[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1

            regression_batch[index, :, :-1] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :])

        # ignore annotations outside of image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1]     = -1
            directions_batch[index, indices, -1] = -1
            regression_batch[index, indices, -1] = -1

    # print("regression_batch: ", regression_batch.shape)
    # print("labels_batch: ", labels_batch)
    # print("directions_batch: ", directions_batch)

    return regression_batch, labels_batch, directions_batch

class XmlCarsAndTrucksGenerator(Generator):
    def __init__(
        self,
        annotations_dir,
        csv_class_file,
        images_root,
        using_direction=False,
        **kwargs
    ):
        self.images_root = images_root


        self.images_annotations = _load_images_annotations(annotations_dir)

        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(XmlCarsAndTrucksGenerator, self).__init__(
            compute_anchor_targets=(north_south_anchor_targets_bbox if using_direction else anchor_targets_bbox),
            **kwargs
        )

    def size(self):
        """ Size of the dataset.
        """
        return len(self.images_annotations)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """

        return os.path.join(self.images_root, self.images_annotations[image_index]["filename"])

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        annotations = {
            "labels": np.empty((0,)),
            "bboxes": np.empty((0, 4)),
            "directions": np.empty((0,)),
        }

        for image_annotation in self.images_annotations[image_index]["annotations"]:
            annotations["labels"] = np.concatenate((annotations["labels"],
                [self.name_to_label(image_annotation["name"])])
            )

            annotations["bboxes"] = np.concatenate((annotations["bboxes"], [image_annotation["bounding_box"]]))

            annotations["directions"] = np.concatenate((annotations["directions"],
                [self.direction_name_to_label(image_annotation["direction"])]
            ))

        return annotations

    # Extra methods for north/south
    def direction_name_to_label(self, direction):
        if direction == "north":
            return 0
        else:
            return 1

    def label_to_direction_name(self, label):
        if label == 0:
            return "north"
        else:
            return "south"
