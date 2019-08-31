import torch
import torchvision.models as models
import torch.nn as nn
import pickle
import numpy as np
from PIL import Image
import collections
import torchvision.transforms as transforms
import os
import shutil
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import random


class tripletWithSoftmax(nn.Module):
    def __init__(self, num_class, pretrain_model_path, finetune=False):
        super(tripletWithSoftmax, self).__init__()
        self.encode_layer = models.resnet101(pretrained=False)
        in_features = self.encode_layer.fc.in_features
        self.encode_layer.fc = nn.Linear(in_features, 128)
        state_dict = torch.load(pretrain_model_path,map_location='cpu')
        for k, v in state_dict.items():
            print(k)
        print('.'*20)
        for k, v in self.encode_layer.state_dict().items():
            print(k)
        self.encode_layer.load_state_dict(state_dict)
        if not finetune:
            for p in self.parameters():
                p.requires_grad = False
        self.softmax = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.encode_layer(x)
        x = self.softmax(x)
        return x


class classifiyToTriplet(nn.Module):
    def __init__(self, encode_dim, pretrain_model_path, finetune=True):
        super(classifiyToTriplet, self).__init__()
        self.encode_layer = models.resnet50(pretrained=False)
        state_dict = torch.load(pretrain_model_path,map_location='cpu')
        self.encode_layer.load_state_dict(state_dict)
        if not finetune:
            for p in self.parameters():
                p.requires_grad = False
        in_features = self.encode_layer.fc.in_features
        self.encode_layer.fc = nn.Linear(in_features, encode_dim)

    def forward(self, x):
        x = self.encode_layer(x)
        return x


class TripletLossNet(nn.Module):
    def __init__(self, device, encode_dim=128, pretrain_model_path='./TripletLossNet/models/pretrain_model.pth', feature_points='./TripletLossNet/models/newest_model/feature_points.pkl', ind_to_name='./TripletLossNet/models/newest_model/idx_to_name.pkl', idx_to_sku= './TripletLossNet/models/newest_model/idx_to_uuid.pkl', finetune=True):
        '''
        :param encode_dim: the dimension of feature points, 128
        :param pretrain_model_path: the path of pretrained model
        :param feature_points: the path of pretrained feature points, one feature point for each class, n_class * 128
        :param ind_to_name: dictionary that map label(int) to real class name
        :param device: define which GPU to run
        :param finetune:
        '''
#         model = TripletLossNet(128, './TripletLossNet/models/pretrain_model.pth', './TripletLossNet/models/newest_model/feature_points.pkl', './TripletLossNet/models/newest_model/idx_to_name.pkl', './TripletLossNet/models/newest_model/idx_to_uuid.pkl', device)
        super(TripletLossNet, self).__init__()
        self.encode_dim = encode_dim
        self.encode_layer = models.resnet50(pretrained=False)
        in_features = self.encode_layer.fc.in_features
        self.encode_layer.fc = nn.Linear(in_features, encode_dim)
        state_dict = torch.load(pretrain_model_path,map_location='cpu')
        state = {}
        for k, v in state_dict.items():
            state[k[13:]] = v
        self.encode_layer.load_state_dict(state)

        self.device = device
        class_info = open(ind_to_name, 'rb')
        self.ind_to_name = pickle.load(class_info)
        class_info.close()

        file = open(feature_points, 'rb')
        self.feature_points = pickle.load(file)
        file.close()
        
        file = open(idx_to_sku, 'rb')
        self.idx_to_sku = pickle.load(file)
        file.close()

    def forward(self, x):
        self.encode_layer.to(self.device)
        x = self.encode_layer(x)
        return x

    def predict(self, x):
        x = self.transform(x)
        x = x.to(self.device)
        calc_representation = self.forward(x).data.cpu().numpy()
#         print(calc_representation)
        distances = np.sqrt(((self.feature_points - calc_representation) ** 2).sum(axis=1))
        prob = np.exp(-distances) / np.exp(-distances).sum()
#         print(prob)
        pred_label = np.argmax(prob, axis=0)
        pred_class_name = self.ind_to_name[pred_label]
        return pred_class_name
    
    def predict_top(self, x, top=3):
        x = self.transform(x)
        x = x.to(self.device)
        calc_representation = self.forward(x).data.cpu().numpy()
#         print(calc_representation)
#         distances = np.sqrt(((self.feature_points - calc_representation) ** 2).sum(axis=1))
        distances = (((self.feature_points - calc_representation) ** 2).sum(axis=1))    
        prob = np.exp(-distances) / np.exp(-distances).sum()
#         print(prob.shape)
        pred_label_list = np.argsort(prob.reshape(1, -1))
        pred_class_name = []
        pred_class_prob = []
        pred_class_sku = []
        for i in range(top):
            pred_class_name.append(self.ind_to_name[pred_label_list[0][-1-i]])
            pred_class_prob.append(distances[pred_label_list[0][-1-i]])
            pred_class_sku.append(self.idx_to_sku[pred_label_list[0][-1-i]])
            
        result = {'name': pred_class_name, 'prob': pred_class_prob, 'sku': pred_class_sku}
        return result
    
    def retrain(self, pic_list, sku_name, uuid):
        batch = torch.ones(1, 3, 150, 150)
        for pic_path in pic_list:
            enforce_pic_list = image_enforce(pic_path)
            for enforce_pic in enforce_pic_list:
                enforce_pic = transforms.ToTensor()(enforce_pic)
                enforce_pic = enforce_pic.unsqueeze(0)
                batch = torch.cat((batch, enforce_pic), 0)
        batch = batch[1:]
        batch = batch.to(self.device)
        calc_new_class_repre = self.forward(batch).mean(dim=0)
        distances = ((self.feature_points - calc_new_class_repre.data.cpu().numpy().reshape(1, self.encode_dim))**2).sum(axis=1)
        if np.min(distances) < 3:
            index = np.argmin(distances)
            info = {'msg': 'fail', 'similar_object_name': self.ind_to_name[index], 'similar_object_sku': self.idx_to_sku[index]}
            return info
        self.feature_points = np.vstack((self.feature_points, calc_new_class_repre.data.cpu().numpy().reshape(1, self.encode_dim)))
        num_class = len(self.ind_to_name)
        num_class += 1
        self.ind_to_name[num_class-1] = sku_name
        self.idx_to_sku[num_class-1] = uuid
#         self.save(self.feature_points, './TripletLossNet/models/newest_model/feature_points.pkl')
#         self.save(self.ind_to_name, './TripletLossNet/models/newest_model/idx_to_name.pkl')
#         self.save(self.feature_points, './TripletLossNet/models/back_up/class_%d_feature_points.pkl'%num_class)
#         self.save(self.ind_to_name, './TripletLossNet/models/back_up/idx_to_name_%d.pkl'%num_class)
        print(calc_new_class_repre.data.cpu().numpy())
        info = {'msg': 'success'}
        return info

    def predict_batch(self, pic_list, top=3):
        batch = torch.ones(1, 3, 150, 150)
        for pic_path in pic_list:
            pic = self.transform(pic_path)
            batch = torch.cat((batch, pic), 0)
        batch = batch[1:]
        batch = batch.to(self.device)
        batch_repre = self.forward(batch).data.cpu().numpy()
        result = []
        for repre in batch_repre:
            distances = np.sqrt(((self.feature_points - repre) ** 2).sum(axis=1))
            prob = np.exp(-distances) / np.exp(-distances).sum()
    #         print(prob)
#             pred_label = np.argmax(prob, axis=0)
        
            pred_label_list = np.argsort(prob.reshape(1, -1))
            pred_class_name = []
            pred_class_prob = []
            pred_class_sku = []
            for i in range(top):
                pred_class_name.append(self.ind_to_name[pred_label_list[0][-1-i]])
                pred_class_prob.append(distances[pred_label_list[0][-1-i]])
                pred_class_sku.append(self.idx_to_sku[pred_label_list[0][-1-i]])

            pc = {'name': pred_class_name, 'prob': pred_class_prob, 'sku': pred_class_sku}
        
        
            result.append(pc)

        
#         self.save(self.feature_points, './TripletLossNet/models/newest_model/feature_points.pkl')
#         self.save(self.ind_to_name, './TripletLossNet/models/newest_model/idx_to_name.pkl')
#         self.save(self.feature_points, './TripletLossNet/models/back_up/class_%d_feature_points.pkl'%num_class)
#         self.save(self.ind_to_name, './TripletLossNet/models/back_up/idx_to_name_%d.pkl'%num_class)
        return result
    
    def save(self, obj, save_path):
        file = open(save_path, 'wb')
        pickle.dump(obj, file)
        file.close()

    def transform(self, x):
        x = Image.open(x)
        x = Scale((150, 150))(x)
        x = np.array(x)
        x = transforms.ToTensor()(x)
        x= x.unsqueeze(0)
        return x


class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """


    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation


    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)




# 随机选择图片并进行resize

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
        image = DataAugmentation.randomColor(image)
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
    image = Scale((150, 150))(image)
    enforce_imgs = [DataAugmentation.process_all(image) for _ in range(times)]
    return [image] + enforce_imgs
