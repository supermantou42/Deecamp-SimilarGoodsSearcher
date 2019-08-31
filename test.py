from yolo_tf.yolo import YOLO
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
model = YOLO()

if __name__ == '__main__':
    img = Image.open('uploads/1.jpg')
    # print(img.shape)
    res = model.detect_image(img)
    res.save('tmp.jpg')