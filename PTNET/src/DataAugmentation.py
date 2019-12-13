import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import os, glob

IMAGE_SIZE = (128, 128)

def load_infer_imgs():
    infer_img_dir = "/data/code/lijianying/PrototypicalNet-FSL-PyTorch/infer_img_dir/"
    infer_imgpaths = glob.glob(os.path.join(infer_img_dir,'*.jpg'))
    return infer_imgpaths


class DataAugmentation:
    """
    包含数据增强的八种方式
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(8, 12) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(8, 12) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(8, 12) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(8, 12) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def process_all(image):
        image = DataAugmentation.randomRotation(image)
#         image = DataAugmentation.randomColor(image)
        # image = DataAugmentation.randomGaussian(image)
        return image

    @staticmethod
    def saveImage(image, path):
        image.save(path)


def process_image_channels(image):
    # process the 4 channels .png
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    # process the channel image
    elif image.mode != 'RGB':
        image = image.convert("RGB")
    return image


def image_enforce(image_path, times=4):
    image = DataAugmentation.openImage(image_path)
    image = process_image_channels(image)
    image = image.resize(IMAGE_SIZE)
    enforce_imgs = [DataAugmentation.process_all(image) for _ in range(times)]
    return [image] + enforce_imgs

if __name__ == '__main__':
    img_path_list = load_infer_imgs()
    res= []
    for p in img_path_list[:3]:
        res += [ transforms.ToTensor()(im) for im in image_enforce(p) ]
    import pdb; pdb.set_trace()