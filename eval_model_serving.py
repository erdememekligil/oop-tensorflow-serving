import argparse
import base64
import json
import sys
import grpc
import numpy as np
import requests
import tqdm
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from datasets import create_test_dataset
from utils import numpy_to_bytes


def rest_serving_bytes(x: np.array, serving_path: str) -> np.array:
    png_bytes = numpy_to_bytes(x)  # convert numpy array to png bytes.
    png_base64 = base64.b64encode(png_bytes).decode("UTF-8")

    request_data = {
        "signature_name": "serving_bytes",
        "instances": [
            {
                "b64": png_base64
            }
        ]
    }

    s = requests.Session()
    # uncomment the line below if you are behind a proxy and getting errors.
    # s.trust_env = False
    response = s.post(url=serving_path, json=request_data)
    if response.text is None or "predictions" not in response.text:
        raise ValueError(response)
    response = json.loads(response.text)

    pred = response["predictions"][0]
    return pred


def rest_serving_numpy(x: np.array, serving_path: str) -> np.array:
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    x = x.astype(np.uint8).tolist()

    request_data = {
        "signature_name": "serving_default",
        "instances": x
    }

    s = requests.Session()
    # uncomment the line below if you are behind a proxy and getting errors.
    # s.trust_env = False
    response = s.post(url=serving_path, json=request_data)
    if response.text is None or "predictions" not in response.text:
        raise ValueError(response)
    response = json.loads(response.text)

    pred = response["predictions"]
    if len(pred) == 1:
        pred = pred[0]
    return pred


def grpc_serving_bytes(x: np.array, stub: PredictionServiceStub, model_name: str) -> np.array:
    png_bytes = numpy_to_bytes(x.astype(np.int32))  # convert numpy array to png bytes.

    grpc_request = PredictRequest()
    grpc_request.model_spec.name = model_name
    grpc_request.model_spec.signature_name = 'serving_bytes'

    grpc_request.inputs['image_bytes_string'].CopyFrom(tf.make_tensor_proto(png_bytes))
    result = stub.Predict(grpc_request, 10.0)  # 10 sec timeout

    return tf.make_ndarray(result.outputs["output_0"])


def grpc_serving_numpy(x: np.array, stub: PredictionServiceStub, model_name: str) -> np.array:
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)

    grpc_request = PredictRequest()
    grpc_request.model_spec.name = model_name
    grpc_request.model_spec.signature_name = 'serving_default'

    grpc_request.inputs['input_tensor'].CopyFrom(tf.make_tensor_proto(x, shape=x.shape))
    result = stub.Predict(grpc_request, 10.0)  # 10 sec timeout

    return tf.make_ndarray(result.outputs["output_0"])


def calculate_accuracy_bytes(test_dataset: DatasetV2, rest_path: str = None, stub: PredictionServiceStub = None,
                             model_name: str = None) -> float:
    y_list = []
    pred_list = []

    for batch_x, batch_y in tqdm.tqdm(test_dataset):
        for x, y in zip(batch_x.numpy(), batch_y.numpy()):
            if rest_path is not None:
                prob = rest_serving_bytes(x, serving_path=rest_path)
            else:
                prob = grpc_serving_bytes(x, stub, model_name)
            pred = np.argmax(prob)

            y_list.append(y)
            pred_list.append(pred)

    return accuracy_score(y_list, pred_list)


def calculate_accuracy_numpy(test_dataset: DatasetV2, rest_path: str = None, stub: PredictionServiceStub = None,
                             model_name: str = None) -> float:
    y_list = []
    pred_list = []

    for batch_x, batch_y in tqdm.tqdm(test_dataset):
        if rest_path is not None:
            prob = rest_serving_numpy(batch_x.numpy(), serving_path=rest_path)
        else:
            prob = grpc_serving_numpy(batch_x.numpy(), stub, model_name)
        pred = np.argmax(prob, axis=1)

        y_list.extend(batch_y.numpy())
        pred_list.extend(pred)

    return accuracy_score(y_list, pred_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, required=True,
                        help="Test dataset path. Example: data_directory/test")
    parser.add_argument('--host', type=str, default="localhost",
                        help="Name or IP of the host machine that runs Tensorflow Serving.")
    parser.add_argument('--grpc_port', type=int, default=8500,
                        help="Port for the gRPC API of Tensorflow Serving.")
    parser.add_argument('--rest_port', type=int, default=8501,
                        help="Port for the REST API of Tensorflow Serving.")
    parser.add_argument('--model_name', type=str, default="ResnetModel",
                        help="Model name that is served by Tensorflow Serving.")
    parser.add_argument('--input_width', type=int, default=224,
                        help="Width of the data. The data will be resized if its size is different.")
    parser.add_argument('--input_height', type=int, default=224,
                        help="Height of the the data. The data will be resized if its size is different.")

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    rest_path = f'http://{args.host}:{args.rest_port}/v1/models/{args.model_name}:predict'
    channel = grpc.insecure_channel(f'{args.host}:{args.grpc_port}')
    service_stub = PredictionServiceStub(channel)
    image_shape = (args.input_height, args.input_width)

    test_dataset = create_test_dataset(args.test_dataset_path, image_shape=image_shape)
    print("Test dataset is created.")

    acc = calculate_accuracy_bytes(test_dataset, stub=service_stub, model_name=args.model_name)
    print("GRPC serving_bytes accuracy", acc)

    acc = calculate_accuracy_numpy(test_dataset, stub=service_stub, model_name=args.model_name)
    print("GRPC serving_numpy accuracy", acc)

    acc = calculate_accuracy_bytes(test_dataset, rest_path=rest_path)
    print("REST serving_bytes accuracy", acc)

    acc = calculate_accuracy_numpy(test_dataset, rest_path=rest_path)
    print("REST serving_numpy accuracy", acc)


if __name__ == '__main__':
    main()
