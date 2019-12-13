# coding=utf-8
import torch
from PIL import Image
import numpy as np
import os, io

classes_dir = './PTNET/classes_txt/'
IMAGE_SIZE = (128, 128)

def class_to_idx():
    class_file = os.path.join(classes_dir, 'all_classes.txt')
    classes = get_current_classes(class_file)
    return {name:idx for idx, name in enumerate(classes)}

def idx_to_class():
    class_file = os.path.join(classes_dir, 'all_classes.txt')
    classes = get_current_classes(class_file)
    return {idx:name for idx, name in enumerate(classes)}

def sku_to_name():
    sku_name_file = os.path.join(classes_dir, 'sku_name_file.txt')
    sku_name_dict = {}
    name_sku_dict = {}
    with io.open(sku_name_file,'r',encoding='utf8') as f:
        for line in f.readlines():
            pair = line.strip().split()
            if len(pair) > 2 :
                pair = [ pair[0], ''.join(pair[1:])]
            sku_name_dict[pair[0]] = pair[1]
            name_sku_dict[pair[1]] = pair[0]
    return sku_name_dict,name_sku_dict

def get_current_classes(fname):
    classes = []
    with open(fname,'r') as f:
        for line in f.readlines():
            classes.append(line.strip())
    return classes

def load_img(path):
    x = Image.open(path).convert('RGB')
    x = x.resize(IMAGE_SIZE, Image.ANTIALIAS)
    x = np.array(x, np.float32, copy=False)
    x = np.transpose(x,(2,0,1))   # (3,128,128)
    return x

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)