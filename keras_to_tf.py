import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras


input_file = sys.argv[1]
output_file = sys.argv[2]


model = keras.models.load_model(input_file)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with tf.io.gfile.GFile(output_file, 'wb') as f:
    f.write(tflite_model)


print('done!')