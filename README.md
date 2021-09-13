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
