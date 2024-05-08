from PIL import Image
import os
import glob

def clip(source_dir, target_dir, type, clip_size=256):
    os.makedirs(target_dir, exist_ok=True)

    if type == 'png':
        files = glob.glob(os.path.join(source_dir, '*.png'))
    else:
        files = glob.glob(os.path.join(source_dir, '*.jpg'))

    for file_path in files:
        img = Image.open(file_path)
        width, height = img.size

        num_x = width // clip_size
        num_y = height // clip_size

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        count = 0
        for x in range(num_x):
            for y in range(num_y):
                left = x * clip_size
                upper = y * clip_size
                right = left + clip_size
                lower = upper + clip_size
                cropped_img = img.crop((left, upper, right, lower))

                if type == 'png':
                    cropped_img.save(os.path.join(target_dir, f'{base_name}_{count}.png'))
                else:
                    cropped_img.save(os.path.join(target_dir, f'{base_name}_{count}.jpg'))
                count += 1


def process_directory(source_base, target_base, sub_folders):
    for folder in sub_folders:
        image_dir = os.path.join(source_base, folder, 'image')
        label_dir = os.path.join(source_base, folder, 'label')

        target_image_dir = os.path.join(target_base, folder, 'image')
        target_label_dir = os.path.join(target_base, folder, 'label')

        os.makedirs(target_image_dir, exist_ok=True)
        os.makedirs(target_label_dir, exist_ok=True)

        # 裁剪图片和标签
        clip(image_dir, target_image_dir, 'jpg')
        clip(label_dir, target_label_dir,'png')


source_base = '/public/zjj/public/zjj/xzx/data-road'
target_base = '/public/zjj/public/zjj/xzx/data-road-clipped'

sub_folders = ['clean', 'raw']

process_directory(source_base, target_base, sub_folders)

print("--- Done ---")
