import os
import boto3

from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite 


s3_client = boto3.client('s3')

model_bucket = 'lambda-model-deployment-workshop'
model_key = 'clothing-model-v4.tflite'
model_local_path = '/tmp/clothing-model-v4.tflite'

if not os.path.exists(model_local_path):
    s3_client.download_file(model_bucket, model_key, model_local_path)


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def image_to_array(img):
    return np.array(img, dtype='float32')

def tf_preprocessing(x):
    x /= 127.5
    x -= 1.0
    return x

def convert_to_tensor(img):
    x = image_to_array(img)
    batch = np.expand_dims(x, axis=0)
    return tf_preprocessing(batch)



interpreter = tflite.Interpreter(model_path=model_local_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_index = input_details[0]['index']

output_details = interpreter.get_output_details()
output_index = output_details[0]['index']


def predict(img):
    img = prepare_image(img, target_size=(299, 299))
    X = convert_to_tensor(img)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return preds[0]


labels = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

def decode_predictions(pred):
    result = {c: float(p) for c, p in zip(labels, pred)}
    return result


def lambda_handler(event, context):
    print(event)
    img = download_image(event['url'])
    pred = predict(img)
    result = decode_predictions(pred)
    return result