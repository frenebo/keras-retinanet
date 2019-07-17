import argparse
import cv2
import progressbar

import os
import sys

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from ..utils.prediction_func_generator import generate_prediction_func
from .predict_video import get_video_dims, get_video_framerate, videocap_generator

def main():
    parser = argparse.ArgumentParser(description='Predict video with tensorflow graph')
    parser.add_argument("prediction_model", help="Source h5 model path")
    parser.add_argument("csv_classes", type=str, help="CSV file with class names.")
    parser.add_argument("source_video", type=str, help="Source video file to read.")
    parser.add_argument("output_directory", type=str, help="Directory for output video.")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone name")
    parser.add_argument("--score-threshold", default=0.05, type=float, help="Threshold for displaying a result")
    parser.add_argument("--max-detections", default=100, type=int, help="Maximum number of detections to show")
    parser.add_argument("--cpu", type=int, help="Optionally limit to a cpu. Example --cpu 1,2")
    parser.add_argument("--show-frames", type=bool, action="store_true")

    args = parser.parse_args()

    if args.prediction_model.endswith(".h5"):
        raise TypeError("Model should be a .pb file")

    if args.cpu is not None:
        ret = os.system("taskset -p -c {} {}".format(args.cpu, os.getpid()))
        if ret != 0:
            raise Exception("Taskset error")

    pred_func = generate_prediction_func(
        frozen_graph_filename=args.prediction_model,
        backbone_name=args.backbone,
        csv_classes_path=args.csv_classes,
        max_detections=args.max_detections,
        score_threshold=args.score_threshold, # threshold score for showing prediction
    )

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

    if args.show_frames:
        import cv2
        window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

    with progressbar.ProgressBar(max_value=frame_count) as bar:
        for i, raw_image in enumerate(videocap_generator(video_cap)):
            bar.update(i)

            with_detections = pred_func(raw_image)

            output_video.write(with_detections)
            if args.show_frames:
                cv2.imshow("CSI Camera", with_detections)

    if args.show_frames:
        cv2.destroyAllWindows()

    video_cap.release()
    output_video.release()



if __name__ == "__main__":
    main()