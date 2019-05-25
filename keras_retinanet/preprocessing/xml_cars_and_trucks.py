
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

class XmlCarsAndTrucksGenerator(Generator):
    def __init__(
        self,
        annotations_dir,
        csv_class_file,
        images_root,
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

        super(XmlCarsAndTrucksGenerator, self).__init__(**kwargs)

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
