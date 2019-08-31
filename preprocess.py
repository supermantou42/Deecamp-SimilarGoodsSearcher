from PIL import Image
import cv2
import numpy as np

def process_image_channels(image):
    # process the 4 channels .png
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    # process the channel image
    elif image.mode != 'RGB':
        image = image.convert("RGB")
    return image


def process_image_reshape(img, min_side=224):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side - new_h) // 2, (min_side - new_h) // 2, (min_side - new_w) // 2 + 1, (
                min_side - new_w) // 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) // 2 + 1, (min_side - new_h) // 2, (min_side - new_w) // 2, (
                min_side - new_w) // 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) // 2, (min_side - new_h) // 2, (min_side - new_w) // 2, (
                min_side - new_w) // 2
    else:
        top, bottom, left, right = (min_side - new_h) // 2 + 1, (min_side - new_h) // 2, (min_side - new_w) // 2 + 1, (
                min_side - new_w) // 2
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])  # 从图像边界向上,下,左,右扩的像素数目
    # print pad_img.shape
    # cv2.imwrite("after-" + os.path.basename(filename), pad_img)
    return Image.fromarray(cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB))

def preprocess(image_path, size = 224):
    image = Image.open(image_path, mode='r')
    image = process_image_channels(image)
    image = process_image_reshape(image, size)
    return np.array(image)