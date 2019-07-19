import argparse
from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print q.get()    # prints "[42, None, 'hello']"
    p.join()

def main()
    parser = argparse.ArgumentParser(description='Predict Camera input')
    parser.add_argument("prediction_model", type=str, help="Path to TensorRT prediction model")
    # parser.add_argument("")
    parser.add_argument("csv_classes", type=str, help="CSV file with class names.")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone name")
    parser.add_argument("--score-threshold", default=0.05, type=float, help="Threshold for displaying a result")
    parser.add_argument("--max-detections", default=100, type=int, help="Maximum number of detections to show")
    args = parser.parse_args()

    if args.prediction_model.endswith(".h5"):
        raise TypeError("Model should be a .pb file")

if __name__ == "__main__":
    main()