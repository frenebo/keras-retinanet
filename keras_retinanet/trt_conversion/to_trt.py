import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
from tensorflow.python.framework import graph_io

import argparse

def load_graph_def(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def

def load_graph(frozen_graph_filename):
    graph_def = load_graph_def(frozen_graph_filename)

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

with tf.Session() as sess:
    parser = argparse.ArgumentParser(description="Make graph TensorRT optimized")
    parser.add_argument("load_tf_model", help="Source tf model path")
    parser.add_argument("save_trt_model", help="Path to save trt model")

    args = parser.parse_args()

    graph_def = load_graph_def(args.load_tf_model)

    output_names = [
        "ident_boxes/Identity",
        "ident_scores/Identity",
        "ident_labels/Identity",
    ]
    trt_graph = trt.create_inference_graph(
        input_graph_def=graph_def,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='INT8',
        is_dynamic_op=False,
        minimum_segment_size=5
    )

    graph_io.write_graph(trt_graph, ".",
        args.save_trt_model, as_text=False)