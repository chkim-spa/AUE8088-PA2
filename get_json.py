import json
import os
import argparse

def load_class_names(class_names_file):
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def create_kaist_annotation_file(data_file, output_file, class_names):
    annotations = []
    base_dir = os.path.dirname(data_file)
    labels_dir = os.path.join(base_dir, "train/labels")

    with open(data_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            image_path = line.strip()
            # 이미지 파일 이름을 기반으로 주석 파일 경로를 생성합니다.
            label_file = os.path.join(labels_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
            if os.path.exists(label_file):
                with open(label_file, 'r') as lf:
                    label_lines = lf.readlines()
                    for label_line in label_lines:
                        parts = label_line.strip().split(' ')
                        if len(parts) >= 5:  # Ensure there are enough parts to unpack
                            class_id, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
                            annotations.append({
                                "image_id": os.path.splitext(os.path.basename(image_path))[0],
                                "category_id": class_id,
                                "category_name": class_names[class_id],
                                "bbox": [x, y, w, h],
                                "score": 1.0  # Assuming all annotations have a score of 1.0
                            })
                        else:
                            print(f"Skipping invalid label line in file {label_file}: {label_line.strip()}")
            else:
                print(f"Label file does not exist: {label_file}")

    if annotations:
        with open(output_file, 'w') as json_file:
            json.dump(annotations, json_file, indent=4)
        print(f"KAIST annotation file created: {output_file}")
    else:
        print("No annotations found. Output file will not be created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create KAIST annotation JSON file")
    parser.add_argument('--data', type=str, required=True, help="Path to the data file containing image paths")
    parser.add_argument('--output', type=str, required=True, help="Output path for the JSON file")
    parser.add_argument('--class_name', type=str, required=True, help="Path to the class names file")
    args = parser.parse_args()

    class_names = load_class_names(args.class_name)
    create_kaist_annotation_file(args.data, args.output, class_names)


# python get_json.py --output utils/eval/KAIST_annotation.json --data datasets/kaist-rgbt/train-all-04.txt --class_name datasets/kaist-rgbt/kaist-rgbt.names

