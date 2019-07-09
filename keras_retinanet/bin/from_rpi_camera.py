import cv2
import sys
import os
import argparse
import numpy as np
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from .. import models
from .predict_video import get_video_dims, csv_label_to_name_func
from ..utils.image import resize_image
from ..utils.visualization import draw_detections

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


def load_graph_def(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def

def yield_frames(
    cap, # video capture
    graph_def,
    # bbox_model, # model with bbox layers
    backbone_name,
    csv_classes_path,
    max_detections=100,
    score_threshold=0.05, # threshold score for showing prediction
    ):
    # Codec is just a series of jpegs that make up a video


    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(graph_def, name='')

    preprocess_image = models.backbone(backbone_name).preprocess_image
    label_to_name = csv_label_to_name_func(csv_classes_path)

    # boxes_tensor = tf_sess.graph.get_tensor_by_name("prefix/ident_boxes/Identity:0")
    # scores_tensor = tf_sess.graph.get_tensor_by_name("prefix/ident_scores/Identity:0")
    # labels_tensor = tf_sess.graph.get_tensor_by_name("prefix/ident_labels/Identity:0")

    # for op in tf_sess.graph.get_operations():
    #     print(op.name)     # <--- printing the operations snapshot below
    # for op in ts_sess.graph.get_operations():
    #     print (op.name(), op.value())

    raise Exception("Done")

    while True:
        _, raw_image = cap.read()
        # time.sleep(0.2)
        # print(img.shape)
        image        = preprocess_image(raw_image.copy())
        image, scale = resize_image(image)


        feed_dict = {
            "input_1": image
        }

        boxes, scores, labels = tf_sess.run([boxes_tensor, scores_tensor, labels_tensor], feed_dict)
        print("boxes: ", boxes.shape)
        print("scores: ", scores.shape)
        print("labels: ", labels.shape)

        # print("Read and preprocessed image")

        # predicted = bbox_model.predict_on_batch(np.expand_dims(image, axis=0))
        # print("Predicted image")
        # boxes, scores, labels = predicted[:3]

        # boxes /= scale
        # indices = np.where(scores[0, :] > score_threshold)[0]

        # scores = scores[0][indices]
        # scores_sort = np.argsort(-scores)[:max_detections]

        # image_boxes      = boxes[0, indices[scores_sort], :]
        # image_scores     = scores[scores_sort]
        # image_labels     = labels[0, indices[scores_sort]]

        # draw_detections(
        #     raw_image,
        #     image_boxes,
        #     image_scores,
        #     image_labels,
        #     label_to_name=label_to_name)

        # print("Labeled image")

        yield raw_image

def main():
    parser = argparse.ArgumentParser(description='Predict Camera input')
    parser.add_argument("prediction_model", type=str, help="Path to TensorRT prediction model")
    # parser.add_argument("")
    parser.add_argument("csv_classes", type=str, help="CSV file with class names.")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone name")
    parser.add_argument("--score_threshold", default=0.05, type=float, help="Threshold for displaying a result")
    args = parser.parse_args()

    if args.prediction_model.endswith(".h5"):
        raise TypeError("Model should be a .pb file")


    stream_string = gstreamer_pipeline(flip_method=0)
    print(stream_string)

    graph_def = load_graph_def(args.prediction_model)

    cap = cv2.VideoCapture(stream_string, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise Exception("Unable to open camera")

    # model = models.load_model(args.prediction_model, backbone_name=args.backbone)
    # bbox_model = models.convert_model(model, using_direction=False)
    try:
        for image_out in yield_frames(
                cap=cap,
                graph_def=graph_def,
                # bbox_model=bbox_model,
                backbone_name=args.backbone,
                csv_classes_path=args.csv_classes,
                score_threshold=args.score_threshold,
            ):
            cv2.imshow("CSI Camera", image_out)
            print("Displayed image")
            keyCode =  cv2.waitKey(30) & 0xff

            # Stop the program on the ESC key
            if keyCode == 27:
                break
    except:
        cap.release()
        cv2.destroyAllWindows()
        raise



    # window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

    # while cv2.getWindowProperty('CSI Camera',0) >= 0:
    #     ret_val, img = cap.read();
    #     cv2.imshow('CSI Camera',img)
    #     keyCode = cv2.waitKey(30) & 0xff

    #     # Stop the program on the ESC key
    #     if keyCode == 27:
    #         break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()