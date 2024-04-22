import os
import base64
from flask import Flask, render_template,request
import config
#newly added
import json
from google.cloud import storage
from google.oauth2 import service_account
#local imports
from detection_model_run import run_detection
from helper import preprocess_keypoints
from classification_model_run import run_classification
from generate_light import  generate_new_image
from show_points import display_keypoints


app = Flask(__name__)
app.config.from_object(config)

UPLOAD_FOLDER = 'captures'  # Define the directory to save uploaded images

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    return render_template("index.html")


def run_model_evaluation(image_path, useGan=False, imageID=0):
    # run keypoint detection on image from camera 
   
    if useGan:
        new_image_path = generate_new_image(image_path, app.config['GAN_MODEL_WEIGHTS_PATH'], imageID = imageID)
        keypoints= run_detection(new_image_path, app.config['POSE_MODEL_WEIGHTS_PATH_GAN'])
        if isinstance(keypoints, str):
            display_keypoints(keypoints, ganImage = True, imageID = imageID)
            return 'No keypoints detected'
        display_keypoints(keypoints, ganImage = True, imageID = imageID)
    else:
        keypoints = run_detection(image_path, app.config['POSE_MODEL_WEIGHTS_PATH_NOGAN'])
        if isinstance(keypoints, str):
            display_keypoints(keypoints, ganImage = False, imageID = imageID)
            return 'No keypoints detected'
        display_keypoints(keypoints, ganImage = False, imageID = imageID)
    #preprocess the keypoints for classification
    input_array = preprocess_keypoints(keypoints)
    predicted_class = run_classification(input_array)
    categoryOrder = ['basketball', 'bowling', 'boxing', 'football', 'golf', 'hacky sack',
       'rowing, stationary', 'skateboarding', 'skiing, downhill', 'soccer',
       'softball, general',
       'tennis, hitting balls, non-game play, moderate effort']

    return categoryOrder[predicted_class]
   

@app.route('/upload', methods=['POST'])
def upload():
    data_url = request.json.get('image_data')
    useGAN = request.json.get('use_model_gan')
    imageID = request.json.get('unique_ID')
    if data_url:
        # Remove header from base64 encoded image
        img_data = data_url.split(',')[1]


        
        # Save the image to a file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'image_{imageID}.png')
        with open( image_path,'wb') as f:
            f.write(base64.b64decode(img_data))

        if useGAN == False:
            answer = run_model_evaluation(image_path, useGan = False, imageID = imageID)
        else:
            answer = run_model_evaluation(image_path, useGan = True, imageID= imageID)
        return answer
    return 'No image data received.'


if __name__=="__main__":
    app.run()

