# tflite_obj_detection
TF lite model to count number of humans in a space.


## usage
- Get the model and its labels. Takes argument of the path, if blank saves in ./models directory
```console
bash ./download.sh ./models
```

- Run the script
```console
python3 main.py \
        --model ./models/detect.tflite \
        --labels ./models/coco_labels.txt \
        --image ./bk.png
```

- If you're running Debian Linux or a derivative of Debian (including Raspberry Pi OS), you should install from our Debian package repo. This requires that you add a new repo list and key to your system and then install as follows:
```console
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime
```
