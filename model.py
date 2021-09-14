import tflite_runtime.interpreter as tflite
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import time
import numpy as np
from pprint import pprint
import re
import argparse


def draw_bounding_box_on_image( image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (
        xmin * im_width,
        xmax * im_width,
        ymin * im_height,
        ymax * im_height,
    )
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=20, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype(
            "./LiberationSansNarrow-Regular.ttf", 25
        )
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(len(boxes), max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(
                class_names[i], int(100 * scores[i])
            )
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str],)
            np.copyto(image, np.array(image_pil))
    return image_pil


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r"[:\s]+", content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details["index"]))
    return tensor


def count_humans(humans):
    count = 0
    for human in humans:
        if human["class_id"] == "person":
            count += 1
    return count


def detect_objects(interpreter, image, threshold ,labels):
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
                "bounding_box": boxes[i],
                "class_id": labels[classes[i]],
                "score": scores[i],
            }
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='File path of .tflite file.', required=True)
    parser.add_argument('--labels', help='File path of labels file.', required=True)
    parser.add_argument('--image', help='File path of the image on to detection is to be performed.', required=True)

    args = parser.parse_args()
    labels = load_labels(args.labels)
    interpreter = tflite.Interpreter(args.model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]["shape"]

    image = Image.open(args.image).convert("RGB").resize((width, height), Image.ANTIALIAS)

    start_time = time.monotonic()
    results = detect_objects(interpreter, image, 0.1, labels)
    elapsed_ms = (time.monotonic() - start_time) * 1000

    print("Inference Time : ", elapsed_ms)
    total_humans = count_humans(results)

    all_scores = []
    all_classes = []
    all_boxes = []
    for result in results:
        all_scores.append(result["score"])
        all_classes.append(result["class_id"])
        all_boxes.append(result["bounding_box"])

    image = np.array(image)
    image_with_boxes = draw_boxes( image, all_boxes, all_classes, all_scores)
    image_with_boxes.save("./labeled_photo.png")

    pprint(results)

    print("Total humans -> ", total_humans)


def get_humans(image_path):
    """
    Returns a int of number humans given the image path.
    """
    labels = load_labels("./models/coco_labels.txt")
    interpreter = tflite.Interpreter("./models/detect.tflite")
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]["shape"]

    image = Image.open(image_path).convert("RGB").resize((width, height), Image.ANTIALIAS)

    start_time = time.monotonic()
    results = detect_objects(interpreter, image, 0.1, labels)
    elapsed_ms = (time.monotonic() - start_time) * 1000

    print("Inference Time : ", elapsed_ms)
    total_humans = count_humans(results)

    all_scores = []
    all_classes = []
    all_boxes = []
    for result in results:
        all_scores.append(result["score"])
        all_classes.append(result["class_id"])
        all_boxes.append(result["bounding_box"])

    image = np.array(image)
    image_with_boxes = draw_boxes( image, all_boxes, all_classes, all_scores)
    image_with_boxes.save("./labeled_photo.png")

    pprint(results)

    print("Total humans -> ", total_humans)

    return total_humans

if __name__ == "__main__":
    main()