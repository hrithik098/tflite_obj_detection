
from flask import request, jsonify, Flask
import numpy as np
import cv2
import os
from multiprocessing import Value
import model
import shutil

app = Flask(__name__)


counter = Value("i", 0)


def save_image(img):
    with counter.get_lock():
        counter.value += 1
        count = counter.value
    image_dir = "images"
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # store the image in above path
    cv2.imwrite(os.path.join(image_dir,"img_"+str(count)+".jpg"), img)


@app.route("/", methods=["GET"])
def index():
    data = {"code": 200, "message": "Connection succesfull, Can start sending data."}
    return jsonify(data)


# delete image if there are more than 50 images
def delete_image():
    shutil.rmtree("./images")


@app.route("/tflite/personCount", methods=["POST"])
def upload():
    img = None
    if request.files:

        file = request.files["imageFile"]

         # convert string of image data to uint8
        np_image = np.fromstring(file.read(), np.uint8)

        # decode image

        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # save image in images folder

        save_image(img)

        # run model on image
        total_humans = model.get_humans('./images/img_'
                                    + str(counter.value) + '.jpg')

        # check if there are more than 360 photos delete all photos
        if counter.value > 360:
            deleted = delete_image()
            counter.value = 0
            if deleted:
                print("Deleted images")
            else:
                print("Path not exists")

        data = {'code': 201, 'message': 'Saved Image',
            'person_count': total_humans}
    else:
        data = {"code": 204, "message": "[FAILED] Image Not Received"}

    return jsonify(data)

if __name__ == "__main__":
    app.run(host='0.0.0.0')

