from flask import request, jsonify, Flask
import numpy as np
import cv2
import os
from multiprocessing import Value
import model

app = Flask(__name__)


counter = Value('i', 0)

def save_image(img):
    with counter.get_lock():
        counter.value += 1
        count = counter.value
    image_dir = "images"
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

     # save image in images/img_<count>.jpg format
    file_path = os.path.join(image_dir, "img_" + str(count) + ".jpg")

    # store the image in above path
    cv2.imwrite(os.path.join(file_path, img))
    return file_path


@app.route('/', methods=['GET'])
def index():
    data = {"code": 200, "message": "Connection succesfull, Can start sending data."}
    return jsonify(data)

# delete image if there are more than 50 images
def delete_image(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


@app.route('/tflite/personCount', methods=['POST'])
def upload():
    if request.files:
        
        file  = request.files['imageFile']

        # convert string of image data to uint8
        np_image = np.fromstring(file.read(), np.uint8)

        # decode image
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # save image in images folder
        file_path = save_image(img)

        # run model on image
        total_humans = model.get_humans(file_path)

        
        data = {"code": 201, "message": "Saved Image", "person_count": total_humans}

    else:
        data = {"code": 204, "message": "[FAILED] Image Not Received"}
    
    return jsonify(data)

app.run(host='0.0.0.0', port=5000)