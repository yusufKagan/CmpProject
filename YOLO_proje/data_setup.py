import xml.etree.ElementTree as ET
import os
import random
import shutil

def xml_to_yolo(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    yolo_labels = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        if class_name == "mask_weared_incorrect":
            class_id = 0
        elif class_name == "without_mask":
            class_id = 1
        elif class_name == "with_mask":
            class_id = 2

        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    output_file = os.path.join(output_dir, root.find("filename").text.replace(".png", ".txt"))
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_labels))

    print(f"Converted {xml_file} to {output_file}")

def process_all_xml_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".xml"):
            xml_file = os.path.join(input_dir, file_name)
            xml_to_yolo(xml_file, output_dir)

def select_validation_files(input_dir, validation_dir, percentage=10):
    os.makedirs(validation_dir, exist_ok=True) 
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    validation_count = max(1, len(txt_files) * percentage // 100)
    validation_files = random.sample(txt_files, validation_count)

    for file_name in validation_files:
        src_path = os.path.join(input_dir, file_name)
        dst_path = os.path.join(validation_dir, file_name)
        shutil.move(src_path, dst_path) 

    print(f"Moved {len(validation_files)} files to validation directory {validation_dir}")

def find_matching_files(png_dir, txt_dir):
    png_files = {os.path.splitext(f)[0] for f in os.listdir(png_dir) if f.endswith('.png')}
    txt_files = {os.path.splitext(f)[0] for f in os.listdir(txt_dir) if f.endswith('.txt')}
    matching_files = png_files.intersection(txt_files)

    return matching_files

def copy_matching_files(png_dir, txt_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    matching_files = find_matching_files(png_dir, txt_dir)

    for file in matching_files:
        png_path = os.path.join(png_dir, file + ".png")
        if os.path.exists(png_path):
            shutil.move(png_path, output_dir)  

def copy_all_png_files(png_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(png_dir):
        if file.endswith('.png'):
            shutil.move(os.path.join(png_dir, file), output_dir) 

input_dir = "archive/annotations"
input_image_dir = "archive/images"

my_directory = ""

output_dir = os.path.join(my_directory, "datasets")
output_dir = os.path.join(output_dir, "second_output_files")
os.makedirs(output_dir, exist_ok=True)

image_folder = os.path.join(output_dir, "images")
os.makedirs(image_folder, exist_ok=True)

train_image = os.path.join(image_folder, "train")
os.makedirs(train_image, exist_ok=True)

val_image = os.path.join(image_folder, "val")
os.makedirs(val_image, exist_ok=True)

label_folder = os.path.join(output_dir, "labels")
os.makedirs(label_folder, exist_ok=True)

train_label = os.path.join(label_folder, "train")
os.makedirs(train_label, exist_ok=True)

val_label = os.path.join(label_folder, "val")
os.makedirs(val_label, exist_ok=True)

process_all_xml_files(input_dir, train_label)
select_validation_files(train_label, val_label)
copy_matching_files(input_image_dir, val_label, val_image)
copy_matching_files(input_image_dir, train_label, train_image)
