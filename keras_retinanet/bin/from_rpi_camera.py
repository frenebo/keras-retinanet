import argparse
import cv2
import sys
import os

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from ..utils.prediction_func_generator import generate_prediction_func

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :
    return (
        'nvarguscamerasrc ! '
        'video/x-raw(memory:NVMM), '
        'width=(int)%d, height=(int)%d, '
        'format=(string)NV12, framerate=(fraction)%d/1 ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink'  % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height
        )
    )

def show_camera():
    parser = argparse.ArgumentParser(description='Predict Camera input')
    parser.add_argument("prediction_model", type=str, help="Path to TensorRT prediction model")
    # parser.add_argument("")
    parser.add_argument("csv_classes", type=str, help="CSV file with class names.")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone name")
    parser.add_argument("--score-threshold", default=0.05, type=float, help="Threshold for displaying a result")
    parser.add_argument("--max-detections", default=100, type=int, help="Maximum number of detections to show")
    parser.add_argument("--output-directory", help="Store to a video file in directory instead of displaying")
    args = parser.parse_args()

    if args.prediction_model.endswith(".h5"):
        raise TypeError("Model should be a .pb file")

    pred_func = generate_prediction_func(
        frozen_graph_filename=args.prediction_model,
        backbone_name=args.backbone,
        csv_classes_path=args.csv_classes,
        max_detections=args.max_detections,
        score_threshold=args.score_threshold, # threshold score for showing prediction
    )

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    # print(gstreamer_pipeline(flip_method=0))

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if args.output_directory is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        framerate = 1.0 # arbitrary
        output_video = cv2.VideoWriter(
            os.path.join(args.output_directory, "camera_output.avi"),
            fourcc,
            framerate,
            get_video_dims(video_cap)
        )

    if cap.isOpened():
        if args.output_directory is None:
            window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
            # Window
            while cv2.getWindowProperty('CSI Camera',0) >= 0:
                _, img = cap.read()

                labeled_img = pred_func(img)

                print("Showing image... ", end="")
                cv2.imshow('CSI Camera', labeled_img)
                print("Done showing image")

                keyCode = cv2.waitKey(30) & 0xff

                if keyCode == 27:
                    break
        else:
            try:
                while True:
                    _, img = cap.read()
                    labled_img = pred_func(img)
                    output_video.write(with_detections)
            except KeyboardInterrupt:
                output_video.release()

        cap.release()
        cv2.destroyAllWindows()
    else:
        print('Unable to open camera')


if __name__ == '__main__':
    show_camera()
