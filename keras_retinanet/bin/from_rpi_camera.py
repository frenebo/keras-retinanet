import cv2
import sys
import os
import argparse

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from .. import models
from .predict_video import get_video_dims

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def yield_frames(
    capture, # video capture
    prediction_model, # model with bbox layers
    ):
    # Codec is just a series of jpegs that make up a video

    while True:
        ret_val, img = cap.read()
        print(img)

        yield img

def main():
    parser = argparse.ArgumentParser(description='Predict video')
    parser.add_argument("prediction_model", type=str, help="Path to prediction model")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone name")
    parser.add_argument("csv_classes", type=str, help="CSV file with class names.")
    parser.add_argument("--score_threshold", default=0.05, type=float, help="Threshold for displaying a result")
    args = parser.parse_args()

    model = models.load_model(args.prediction_model, backbone_name=args.backbone)
    bbox_model = models.convert_model(model, using_direction=args.using_direction)

    stream_string = gstreamer_pipeline(flip_method=0)
    print(stream_string)
    cap = cv2.VideoCapture(stream_string)

    if not cap.isOpened():
        raise Exception("Unable to open camera")

    for image_out in yield_frames(
            capture=cap,
            prediction_model=bbox_model,
        ):
        cv2.imshow("CSI Camera", image_out)
        keyCode =  cv2.waitKey(30) & 0xff

        # Stop the program on the ESC key
        if keyCode == 27:
            break




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