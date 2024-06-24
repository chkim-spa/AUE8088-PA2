import json
import os
import cv2
import argparse
import matplotlib.pyplot as plt

def draw_labels_on_image(image_path, annotations, output_dir):
    image = cv2.imread(image_path)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        category = ann["category_name"]
        
        # Convert normalized coordinates to pixel coordinates
        img_height, img_width, _ = image.shape
        x = int(x * img_width)
        y = int(y * img_height)
        w = int(w * img_width)
        h = int(h * img_height)

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Put the label above the bounding box
        cv2.putText(image, category, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Save the image with labels
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

def load_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw labels on images")
    parser.add_argument('--images-dir', type=str, required=True, help="Path to the directory containing images")
    parser.add_argument('--annotations', type=str, required=True, help="Path to the JSON file containing annotations")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save images with labels")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    image_annotations = {}
    for ann in annotations:
        image_id = ann["image_id"]
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

    for image_id, anns in image_annotations.items():
        image_path = os.path.join(args.images_dir, f"{image_id}.jpg")
        if os.path.exists(image_path):
            draw_labels_on_image(image_path, anns, args.output_dir)
        else:
            print(f"Image file does not exist: {image_path}")

# python visual_json.py --images-dir datasets/kaist-rgbt/train/images/visible --annotations utils/eval/KAIST_annotation.json --output-dir datasets/visual_json
