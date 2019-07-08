import argparse
import cv2
import csv
import progressbar
import os
import sys

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from ..preprocessing.generator import Generator
from .. import models
from ..utils.image import resize_image
from ..utils.visualization import draw_detections
from ..preprocessing.csv_generator import _read_classes


def get_video_framerate(video_capture):
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print("Input video frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    return fps

def get_video_dims(vid_capture):
    width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (width, height)

def videocap_generator(video_capture):
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        yield frame


def bbox_detect_image(
    bbox_model,
    backbone_name,
    raw_image,
    label_to_name, # Should be label_to_name from generator
    score_threshold=0.15,
    max_detections=100,
    using_direction=False,
    label_to_direction_name=None, # Should also be from generator
    print_averages=False,
    ):
    preprocess_image = models.backbone(backbone_name).preprocess_image

    image        = preprocess_image(raw_image.copy())
    image, scale = resize_image(image)

    predicted = bbox_model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes, scores, labels = predicted[:3]
    if using_direction:
        directions = predicted[3]
        # print("labels shape: ", labels.shape)
        # print("labels dtype: ", labels.dtype)
        # print("directions shape: ", directions.shape)
        # print("directions dtype: ", directions.dtype)

    boxes /= scale
    indices = np.where(scores[0, :] > score_threshold)[0]

    if print_averages:
        print("Average score: ", np.average(scores))

    # print("scores: ", scores)
    # print("indices: ", indices)

    scores = scores[0][indices]
    scores_sort = np.argsort(-scores)[:max_detections]

    image_boxes      = boxes[0, indices[scores_sort], :]
    image_scores     = scores[scores_sort]
    image_labels     = labels[0, indices[scores_sort]]
    if using_direction:
        image_directions = directions[0, indices[scores_sort]]

    labeled_image = raw_image.copy()

    draw_detections(
        labeled_image,
        image_boxes,
        image_scores,
        image_labels,
        label_to_name=label_to_name,
        using_direction=using_direction,
        directions=image_directions if using_direction else None,
        label_to_direction_name=label_to_direction_name)

    return labeled_image

def csv_label_to_name_func(csv_classes_path):

    with open(csv_classes_path, 'r', newline='') as csv_file:
        classes = _read_classes(csv.reader(csv_file, delimiter=','))

    labels =  {}
    for key, value in classes.items():
        labels[value] = key

    def label_to_name(label):
        return labels[label]

    return label_to_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict video')
    parser.add_argument("prediction_model", type=str, help="Path to prediction model")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone name")
    parser.add_argument("csv_classes", type=str, help="CSV file with class names.")
    parser.add_argument("source_video", type=str, help="Source video file to read.")
    parser.add_argument("output_directory", type=str, help="Directory for output video.")
    parser.add_argument("--cuda_visible_devices", default=None, type=str, help="Optional CUDA visible devices param to limit which gpus are used.")
    parser.add_argument("--score_threshold", default=0.05, type=float, help="Threshold for displaying a result")
    parser.add_argument("--using_direction", action='store_true', help="Whether or not to use direction")

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


    # model = load_retinanet(args.prediction_model)
    model = models.load_model(args.prediction_model, backbone_name=args.backbone)
    bbox_model = models.convert_model(model, using_direction=args.using_direction)
    # bbox_model = create_bbox_retinanet(model, anchor_config, using_direction=args.using_direction)

    video_cap = cv2.VideoCapture(args.source_video)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Codec is just a series of jpegs that make up a video
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    framerate = get_video_framerate(video_cap)
    output_video = cv2.VideoWriter(
        os.path.join(args.output_directory, "output.avi"),
        fourcc,
        framerate,
        get_video_dims(video_cap)
    )

    # @TODO find cleaner method
    if args.using_direction:
        def stand_in_label_to_direction_name(direction):
            if direction == 0:
                return "north"
            else:
                return "south"
    else:
        stand_in_label_to_direction_name = None

    label_to_name = csv_label_to_name_func(args.csv_classes)

    with progressbar.ProgressBar(max_value=frame_count) as bar:
        for i, raw_image in enumerate(videocap_generator(video_cap)):
            bar.update(i)

            with_detections = bbox_detect_image(
                bbox_model,
                args.backbone,
                raw_image,
                label_to_name=label_to_name,
                score_threshold=args.score_threshold,
                using_direction=args.using_direction,
                label_to_direction_name=stand_in_label_to_direction_name)


            # from height, width, depth to width, height, depth
            # with_detections = with_detections.transpose(1, 0, 2)
            # with_detections = cv2.cvtColor(with_detections, cv2.COLOR_RGB2BGR)

            # print(with_detections.shape)
            # print(get_video_dims(video_cap))

            # output_video.write(with_detections)
            output_video.write(with_detections)

        # output_video.

    # while video_cap.isOpened():
    #     ret, frame = video_cap.read()

    #     if not ret:
    #         break

    #     retinanet.predict_image()



    video_cap.release()
    output_video.release()
