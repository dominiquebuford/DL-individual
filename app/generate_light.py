import numpy as np
import cv2, os
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, save_img, array_to_img


def generate_new_image(image_path, model_path, imageID = 0):
    # Processing Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_height, original_width = img.shape[:2]
    img_arr = (img_to_array(img) - 127.5) / 127.5
    resized = cv2.resize(img_arr, (256, 256), interpolation=cv2.INTER_AREA)
    ready_img = np.expand_dims(resized, axis=0)

    # Loading Model
    model = load_model(model_path)

    # Prdicting Image
    pred = model.predict(ready_img)
    pred = (cv2.medianBlur(pred[0], 1) + 1) / 2
    pred = cv2.resize(pred, (original_width, original_height))
    pred = array_to_img(pred)
    final_image_path = f"captures/new_image_{imageID}.png"
    save_img(final_image_path, pred)
    return final_image_path
