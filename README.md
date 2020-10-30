# Tensorflow Lite Object Detection with the Tensorflow Object Detection API

[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)

![Object Detection Example](doc/object_detection_with_edgetpu.png)

# 1.Train a object detection model using the Tensorflow OD API

For this guide you can either use a pre-trained model from the [Tensorflow Model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) or you can train your own custom model as described in [one of my other Github repositories](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model/tree/tf1).

# 2.Convert the model to Tensorflow Lite

After you have a Tensorflow OD model you can start to convert it to Tensorflow Lite.

This is a two-step process:
1. Export frozen inference graph for TFLite
2. Using the *tflite_convert* to create a optimized TensorFlow Lite Model

## 2.1 Export frozen inference graph for TFLite

After training the model you need to export the model so that the graph architecture and network operations are compatible with Tensorflow Lite. This can be done with the ```export_tflite_ssd_graph.py``` file.

To make these commands easier to run, let’s set up some environment variables:

```bash
export CONFIG_FILE=PATH_TO_BE_CONFIGURED/pipeline.config
export CHECKPOINT_PATH=PATH_TO_BE_CONFIGURED/model.ckpt-XXXX
export OUTPUT_DIR=/tmp/tflite
```

on Windows use ```set``` instead of ```export```:

```bash
set CONFIG_FILE=PATH_TO_BE_CONFIGURED/pipeline.config
set CHECKPOINT_PATH=PATH_TO_BE_CONFIGURED/model.ckpt-XXXX
set OUTPUT_DIR=C:<path>/tflite
```

XXXX represents the highest number.

```bash
python export_tflite_ssd_graph.py \
    --pipeline_config_path=$CONFIG_FILE \
    --trained_checkpoint_prefix=$CHECKPOINT_PATH \
    --output_directory=$OUTPUT_DIR \
    --add_postprocessing_op=true
```

In the ```OUTPUT_DIR``` you should now see two files: tflite_graph.pb and tflite_graph.pbtxt

## 2.2 Convert to TFLite

To convert the frozen graph to Tensorflow Lite we need to run it through the Tensorflow Lite Converter. It converts the model into an optimized FlatBuffer format that runs efficiently on Tensorflow Lite.

### 2.2.1 Create Tensorflow Lite model

If you want to convert a quantized model you can run the following command:

```bash
tflite_convert \
    --input_file=$OUTPUT_DIR/tflite_graph.pb \
    --output_file=$OUTPUT_DIR/detect.tflite \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_values=128 \
    --change_concat_input_ranges=false \
    --allow_custom_ops
```

If you are using a floating point model you need to change the command:

```bash
tflite_convert \
    --input_file=$OUTPUT_DIR/tflite_graph.pb \
    --output_file=$OUTPUT_DIR/detect.tflite \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    --inference_type=FLOAT  \
    --allow_custom_ops
```

If things ran successfully, you should now see a third file in the /tmp/tflite directory called detect.tflite.

### 2.2.2 Create new labelmap for Tensorflow Lite

Next you need to create a label map for Tensorflow Lite, since it doesn't have the same format as a classical Tensorflow labelmap.

Tensorflow labelmap:

```bash
item {
    name: "a"
    id: 1
    display_name: "a"
}
item {
    name: "b"
    id: 2
    display_name: "b"
}
item {
    name: "c"
    id: 3
    display_name: "c"
}
```

The Tensorflow Lite labelmap format only has the display_names (if there is no display_name the name is used).

```bash
a
b
c
``` 

So basically the only thing you need to do is to create a new labelmap file and copy the display_names (names) from the other labelmap file into it.

### 2.2.3 Optional: Convert Tensorflow Lite model to use with the Google Coral EdgeTPU

If you want to use the model with a Google Coral EdgeTPU you need to run it through the EdgeTPU Compiler. 

The compiler can be installed on Linux systems (Debian 6.0 or higher) with the following commands:

```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt-get update

sudo apt-get install edgetpu-compiler
```

After installing the compiler you can convert the model with the following command:

```
!edgetpu_compiler --out_dir <output_directory> <path_to_tflite_file>
```

Before using the compiler, be sure you have a model that's compatible with the Edge TPU. For compatibility details, read [TensorFlow models on the Edge TPU](https://coral.ai/docs/edgetpu/models-intro/).

## 3. Using the model for inference

This repository contains two scripts to run the model. On for running the object detection model on a video and one for running it on a webcam. Both can be run with or without the EdgeTPU.

* [Object Detection on video](tflite_object_detection_with_video.py)
* [Object Detection on webcam](tflite_object_detection_with_webcam.py)

## Author
 **Gilbert Tanner**
 
## Support me

<a href="https://www.buymeacoffee.com/gilberttanner" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>