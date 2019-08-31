# coding=utf-8
# from PTNET.src.utils import load_img, euclidean_dist, class_to_idx, idx_to_class, sku_to_name
# from PTNET.src.protonet import ProtoNet
from utils import load_img, euclidean_dist, class_to_idx, idx_to_class, sku_to_name
from protonet import ProtoNet
from DataAugmentation import image_enforce
import torch
import torchvision.transforms as transforms
import numpy as np
import os , glob , io
import pickle


  
def load_infer_imgs():
    infer_img_dir = "/data/code/lijianying/PrototypicalNet-FSL-PyTorch/infer_img_dir/"
    infer_imgpaths = glob.glob(os.path.join(infer_img_dir,'*.jpg'))
    return infer_imgpaths


class PTNET(object):
    
#     new_prototype_label = 200

    def __init__(self):
        super(PTNET, self).__init__()
        self.root = '/data/code/supermantou/flask/PTNET'
        self.model_dir = os.path.join(self.root,'models')
        self.model_path = os.path.join(self.model_dir, 'last_model.pth')
        self.prototypes_path = os.path.join(self.model_dir,'BaseModel_prototype_lable_class100_smps1000.pickle')
        
        self.prototypes, self.labels = self.load_prototypes()  # self.prototypes：torch.Tensor   self.labels: numpy.ndarray  
        self.idx2sku = idx_to_class()
        self.sku2idx = class_to_idx()
        self.sku2name, self.name2sku = sku_to_name()
        self.device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
        
        # 添加新点，起始编号
        self.labelID = 200
        # 剔除某些原型  【缺陷：标签名字相同的多个类，如农夫山泉，目前只能删掉一个且不确定】
        Exclude_PrototypeXs = True
        cls_names = ['雀巢咖啡系列268ml','统一绿茶茉莉味500ml']   # class names of Exclude_PrototypeXs  #康师傅绿茶蜂蜜茉莉低糖500ml
        if Exclude_PrototypeXs:
            for name in cls_names:
                labelID = self.sku2idx[self.name2sku[name]]
                mask = self.labels!=labelID
                self.prototypes = self.prototypes[torch.tensor(mask)]
                self.labels = self.labels[mask]
                

    def load_prototypes(self):
        with open(self.prototypes_path, 'rb') as f:
            prototypes, labels = pickle.load(f)  #  [num_classes,128]
        labels = labels.numpy()
        return prototypes, labels
    

    def predict_batch(self, img_paths_list, top_k):
        # load inference samples
        infer_imgs = list()
        for path in img_paths_list:
            infer_imgs.append( torch.tensor(load_img(path)) )  # list of tensor
        X = torch.stack(infer_imgs)

        # load model
        model = ProtoNet().cpu()
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        model.eval()

        # start inferring
        pred_label_list = list()
        pred_class_name = list()
        pred_class_sku = list()
        pred_class_prob = list()
    
        model_output = model(X)   # [batch_size,128]
        dists = euclidean_dist(model_output.to('cpu'), self.prototypes.to('cpu'))  # [batch_size,num_classes]
        dists = dists.data.cpu().numpy()
        sorted_dists = np.sort(dists,axis=1)
        sorted_idxs = np.argsort(dists,axis=1)
        # whether reject
        threshold = 15.0
        mask = sorted_dists < threshold
        
        for i in range(len(infer_imgs)):
            pred_class_prob.append( sorted_dists[i][mask[i]][:top_k].tolist() )
            pred_label_list.append( self.labels[sorted_idxs[i]][mask[i]][:top_k].tolist() )
            pred_class_sku.append( [ self.idx2sku[idx] for idx in pred_label_list[i] ] )
            pred_class_name.append( [ self.sku2name[idx] for idx in pred_class_sku[i] ] )
            
        result = []  # list of dict for each image
        for i in range(len(infer_imgs)):
            cur_img_result = {'name': pred_class_name[i], 'prob': pred_class_prob[i], 'sku': pred_class_sku[i]}
            result.append(cur_img_result)
            
        return result


    def retrain(self, img_paths_list, class_name, sku ):

        self.labelID += 1

        infer_imgs = []
        for p in img_paths_list:
            infer_imgs += [ transforms.ToTensor()(im) for im in image_enforce(p) ]
        X = torch.stack(infer_imgs)

        # load model
        model = ProtoNet().cpu()
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        model.eval()

        # compute new prototype
        model_output = model(X)   # [batch_size,128]        
        batch_prototype = model_output.mean(0)
        batch_prototype = batch_prototype.unsqueeze(0)
        
        # whether fail to map to a distinguishing emmbedding
        threshold = 0.0
        dists = euclidean_dist(batch_prototype.to('cpu'), self.prototypes.to('cpu'))  # [batch_size,num_classes]
        min_dist = torch.min(dists).item()
        if min_dist < threshold:
            index = np.argmin(dists)
            sim_lblid = self.labels[index] 
            info = {'msg': 'fail', 'similar_object_name': self.sku2name[self.idx2sku[sim_lblid]], 'similar_object_sku': self.idx2sku[sim_lblid]}
            return info

        # add new class info
        self.prototypes = torch.cat([self.prototypes,batch_prototype], 0)
        self.labels = np.concatenate( (self.labels,[self.labelID] ), axis=0 )
        self.idx2sku[self.labelID] = sku
        self.sku2name[sku] = class_name
        
        info = {'msg': 'success'}
        return info

    

if __name__ == '__main__':
    
    debug = True
    if debug:
        img_paths_list = load_infer_imgs()
        
    mynet = PTNET()
    res = mynet.predict_batch(img_paths_list,30)
#     state1 = mynet.retrain(img_paths_list,class_name='new1',sku='bbbbb')
#     state2 = mynet.retrain(img_paths_list,class_name='new2',sku='ccccc')
    print("Finish!")
    import pdb;pdb.set_trace()
    