import os
import glob


def rename_files(directory, type):
    files = glob.glob(os.path.join(directory, f'*.{type}'))
    files.sort()

    for i, file_path in enumerate(files):
        new_name = f"{i + 1}.{type}"
        os.rename(file_path, os.path.join(directory, new_name))


dirs = ['/public/zjj/public/zjj/xzx/data-road-clipped/clean/image',
        '/public/zjj/public/zjj/xzx/data-road-clipped/clean/label',
        '/public/zjj/public/zjj/xzx/data-road-clipped/raw/image',
        '/public/zjj/public/zjj/xzx/data-road-clipped/raw/label']

if __name__ == '__main__':
    for d in dirs:
        if 'image' in d:
            rename_files(d, 'jpg')
        else:
            rename_files(d, 'png')
