import os
import shutil

source_dir = '/public/zjj/public/zjj/jy/work1/dataset/croproad/train'
image_dir = '../data-road/raw/image'
label_dir = '../data-road/raw/label'

os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

files = sorted(os.listdir(source_dir))

counter = 0
max_pairs = 1023

base_to_new_id = {}
for file in files:
    if counter > max_pairs:
        break

    if '(2).png' not in file:
        base_name = file.split('_')[0]
        if base_name not in base_to_new_id:
            base_to_new_id[base_name] = counter
            counter += 1
        new_name = f"{base_to_new_id[base_name]}.jpg"
        new_path = os.path.join(image_dir, new_name)
        full_file_path = os.path.join(source_dir, file)
        shutil.copy(full_file_path, new_path)

for file in files:
    if '(2).png' in file:
        base_name = file.split('_')[0]
        if base_name in base_to_new_id:
            new_name = f"{base_to_new_id[base_name]}.png"
            new_path = os.path.join(label_dir, new_name)
            full_file_path = os.path.join(source_dir, file)
            shutil.copy(full_file_path, new_path)

print("--- Done ---")
