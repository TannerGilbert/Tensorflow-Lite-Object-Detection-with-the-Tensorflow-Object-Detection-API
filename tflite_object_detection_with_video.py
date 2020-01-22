# based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/detect_picamera.py
from tflite_runtime.interpreter import Interpreter, load_delegate
import argparse
import time
import cv2
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_image(image, results, labels, size):
    result_size = len(results)
    for idx, obj in enumerate(results):
        print(obj)
        # Prepare image for drawing
        draw = ImageDraw.Draw(image)

        # Prepare boundary box
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * size[0])
        xmax = int(xmax * size[0])
        ymin = int(ymin * size[1])
        ymax = int(ymax * size[1])

        # Draw rectangle to desired thickness
        for x in range( 0, 4 ):
            draw.rectangle((ymin, xmin, ymax, xmax), outline=(255, 255, 0))

        # Annotate image with label and confidence score
        display_str = labels[obj['class_id']] + ": " + str(round(obj['score']*100, 2)) + "%"
        draw.text((box[0], box[1]), display_str, font=ImageFont.truetype("/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf", 20))

        displayImage = np.asarray( image )
        cv2.imshow('Coral Live Object Detection', displayImage)


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def make_interpreter(model_file, use_edgetpu):
    model_file, *device = model_file.split('@')
    if use_edgetpu:
        return Interpreter(
            model_path=model_file,
            experimental_delegates=[
                load_delegate('libedgetpu.so.1',
                {'device': device[0]} if device else {})
            ]
        )
    else:
        return Interpreter(model_path=model_file)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', type=str, required=True, help='File path of .tflite file.')
    parser.add_argument('-l', '--labels', type=str, required=True, help='File path of labels file.')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, required=False, help='Score threshold for detected objects.')
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to video')
    parser.add_argument('-e', '--use_edgetpu', action='store_true', default=False, help='Use EdgeTPU')
    args = parser.parse_args()

    labels = load_labels(args.labels)
    interpreter = make_interpreter(args.model, args.use_edgetpu)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Initialize video stream
    video = cv2.VideoCapture(video)
    time.sleep(1)

    while(video.isOpened()):
        try:
            ret, frame = video.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (input_width, input_height))
            input_data = np.expand_dims(frame_resized, axis=0)
            # Perform inference
            results = detect_objects(interpreter, input_data, args.threshold)

            draw_image(image, results, labels, image.size)

            if( cv2.waitKey( 5 ) & 0xFF == ord( 'q' ) ):
                fps.stop()
                break
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    video.release()
    time.sleep(2)


if __name__ == '__main__':
    main()