import cv2
import numpy as np
import tensorflow as tf
import json

img_size = 64

# Cargar modelo
model = tf.keras.models.load_model('asl_model.h5')

model.summary()

