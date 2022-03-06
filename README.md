# Object Oriented Model Training and Tensorflow Serving
An object oriented (OOP) approach to train Tensorflow models and serve them using Tensorflow Serving.

## Why bother with OOP?
- OOP is good (in general).
- Custom functions can be added to the model. For example, a custom function that makes predictions and returns the most probable class label instead of probabilities. In addition to creating new functions, existing keras functions like _call_ can be overridden to change their functionalities.
- Overriding keras functions makes the model compatible with other classes or functions. For instance, adding _signatures_ (methods that can be called by Tensorflow Serving) by overriding the _save_ method makes it compatible with the ModelCheckpoint. Since the ModelCheckpoint calls the overridden _save_ method to save models, models that are saved by it will have the extra _signatures_.
- Some architecture independent requirements can be met in a base class and different model architectures can be defined in subclasses of that base class. 

## Extending tf.keras.Model
A base class, ImageClassifierModel, is defined to group up common requirements and capabilities of image classifiers. In this case, we need our models to have two different prediction methods:
- **predict_numpy_image**: The default call method. Makes a prediction using 4d array or tensor input (batch, height, width, channel). 
- **predict_bytes_image**: Makes a prediction using a single image's encoded bytes (png, jpeg, bmp, gif).

These methods are _tf.functions_ and they are set as signatures of the model in the overridden _save_ method of tf.keras.Model. These signature definitions can be served by Tensorflow Serving as gRPC or REST services. Furthermore, the tf.functions can be called even after loading the model from the disk. 

LenetModel and ResnetModel are the two subclasses of ImageClassifierModel and they implement the abstract _create_model_io_ method to define the model architectures. ResnetModel is a wrapper of pre-trained ResNet50V2 and it helps fine-tuning.

## Dataset

MNIST dataset is used in this repository since it is a publicly available, relatively small and fast to train. However, any dataset that is given in the format below is accepted by the code. Images should be in the directories named by class names under the train and test directories. The MNIST dataset can be downloaded from [here](https://www.kaggle.com/jidhumohan/mnist-png).

```bash
data_directory/
...train/
......class_a/
.........a_image_1.jpg
.........a_image_2.jpg
......class_b/
.........b_image_1.jpg
.........b_image_2.jpg
...test/
......class_a/
.........a_image_3.jpg
```

## Training & Evaluation

To run these scripts create a new virtual environment or conda environment (python 3.8 is tested). Then use pip to install the requirements. For detailed information about tensorflow installation follow the [website](https://www.tensorflow.org/install).

```bash
pip install -r requirements.txt
```

To train and evaluate a resnet model use the command below. This command assumes that your train and test data are under the "oop-tensorflow-serving/data" directory. The trained model will be saved under the "oop-tensorflow-serving/models/ResnetModel/1" directory. Please inspect the ArgumentParser in train_model.py for optional parameters like model name and input shape.

```bash
python train_model.py --train_dataset_path data/train --test_dataset_path data/test --model_folder_path models
```

If you want to try a model architecture of your own, implement it in _create_model_io_ method of your newly created class (which extends ImageClassifierModel) and pass in the parameter `--model_name`. For example, we can create a LenetModel class that implements the LeNet architecture, import it in the train_model.py, then pass the class name to the command:

```bash
python train_model.py --train_dataset_path data/train --test_dataset_path data/test --model_folder_path models --model_type LenetModel
```

To test an existing model use this command. This script also demonstrates the usage of predict_numpy_image and predict_bytes_image methods:

```bash
python eval_model.py --test_dataset_path data/test --model_path models/ResnetModel/1
```

Alternatively, you can use the provided Dockerfile to build and image and use it to run these scripts:

```bash
docker build --tag oop-tensorflow-serving:1.0 .

docker run --rm --gpus all \
  -v /directory/on/host/oop-tensorflow-serving:/work \
  oop-tensorflow-serving:1.0 /work/train_model.py \
  --train_dataset_path /work/data/train \
  --test_dataset_path /work/data/test \
  --model_folder_path /work/models
```

### SavedModel

Since the models are in SavedModel format, the model signatures can be shown using this command: 

```bash
saved_model_cli show --all --dir models/ResnetModel/1
```

serving_bytes and serving_default signatures and their input/outputs can be seen in this commands output:

```text
signature_def['serving_bytes']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['image_bytes_string'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: serving_bytes_image_bytes_string:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 10)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_tensor'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1, -1, 3)
        name: serving_default_input_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 10)
        name: StatefulPartitionedCall_1:0
  Method name is: tensorflow/serving/predict
```

## Serving

Tensorflow [recommends](https://www.tensorflow.org/tfx/serving/setup) using Docker image for Tensorflow Serving since it is the easiest way to use TF Serving with GPU support. Follow the instructions in this [link](https://www.tensorflow.org/tfx/serving/setup) if you don't have docker and want to install Tensorflow Serving manually. 

The below script creates and runs a Tensorflow Serving container with the given model. Port 8500 is used for the gRPC API and 8501 is used for the REST API.

```bash
docker run -p 8500:8500 -p 8501:8501 -d --name resnet_serving \
  -v /directory/on/host/models:/models \
  -e MODEL_NAME=ResnetModel tensorflow/serving:2.8.0-gpu
```

To check if it's up and running correctly go to this address in a web browser: `http://hostname:8501/v1/models/ResnetModel`. It returns a json like below if it is working correctly:

```json
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```

To call the model using the REST API send a POST request to `http://hostname:8501/v1/models/ResnetModel:predict` with a request body like this:

```json
{
  "signature_name": "serving_bytes", 
  "instances": [{"b64": "base64_encoded_image_bytes"}]
}
```
The service returns activations of the model: 10 numbers (softmax not applied) for 10 classes. For this sample, the model prediction is 2 (the class with the max value).
```json
{"predictions": [[-14.9772987, -6.99252939, 13.5781298, -8.89471, -6.88773823, -4.63609457, 0.168618962, -9.86182785, -2.09211802, -1.32305372]]}
```

The eval_model_serving.py script evaluates the served model by calling the two signatures via gRPC and REST protocols: 
```bash
python eval_model_serving.py --test_dataset_path data/test --host hostname
```

