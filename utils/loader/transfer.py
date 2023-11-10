import numpy as np
import cv2 as cv

# 饱和度变换
def random_hue_saturation_value(image, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        h, s, v = cv.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv.add(v, val_shift)
        image = cv.merge((h, s, v))
        image = cv.cvtColor(image, cv.COLOR_HSV2RGB)
    return image

# 透视变换
def random_shift_scale_rotate(image, mask, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0), rotate_limit=(-0.0, 0.0), aspect_limit=(-0.0, 0.0), border_mode=cv.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv.getPerspectiveTransform(box0, box1)
        image = cv.warpPerspective(image, mat, (width, height), flags=cv.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
        mask = cv.warpPerspective(mask, mat, (width, height), flags=cv.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
    return image, mask

# 水平翻转
def random_horizontal_flip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv.flip(image, 1)
        mask = cv.flip(mask, 1)
    return image, mask

# 垂直翻转
def random_ver_flip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv.flip(image, 0)
        mask = cv.flip(mask, 0)
    return image, mask

# 90°翻转
def random_rotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)
    return image, mask