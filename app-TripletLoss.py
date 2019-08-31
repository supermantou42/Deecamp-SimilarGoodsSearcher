import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
from TripletLossNet.TripletLoss import TripletLossNet
from preprocess import preprocess
import uuid
import shutil
import urllib.request
from werkzeug.utils import secure_filename
import torch
from yolo_tf.yolo import YOLO
from PIL import Image
# import importlib,sys
# importlib.reload(sys) 
# sys.setdefaultencoding('utf8')   


app = Flask(__name__)

url_prefix = '/infer-ed58ae0d-a251-42e0-aa9a-329181df32d6'
root_url = 'https://jupyter-uaitrain-bj2.ucloud.cn' + url_prefix
# url_prefix = '/prefix'
# root_url = 'http://127.0.0.1:5000'

UPLOAD_FOLDER = 'static/uploads'
SKU_URL_PREFIX = root_url + '/static/sku_face/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
# device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
print('basedir:', basedir)

detector = YOLO()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



# model = resnet(device)
# model = TripletLossNet(128, './TripletLossNet/models/pretrain_model.pth', './TripletLossNet/models/newest_model/feature_points.pkl', './TripletLossNet/models/newest_model/idx_to_name.pkl', './TripletLossNet/models/newest_model/idx_to_uuid.pkl', device)
# (self, device, encode_dim=128, pretrain_model_path='./TripletLossNet/models/pretrain_model.pth', feature_points='./TripletLossNet/models/newest_model/feature_points.pkl', ind_to_name='./TripletLossNet/models/newest_model/idx_to_name.pkl', idx_to_sku= './TripletLossNet/models/newest_model/idx_to_uuid.pkl', finetune=True)
# model = TripletLossNet(device, pretrain_model_path='./TripletLossNet/models/data_agm_tripletLoss_class_132_1-Copy1.086_val_loss.pth', feature_points='./TripletLossNet/models/feature_points_agm_pretrain-Copy1.pkl')
# model = TripletLossNet(device, pretrain_model_path='./TripletLossNet/models/STAR-data_agm_tripletLoss_class_132_1-Copy1.076_val_loss.pth', feature_points='./TripletLossNet/models/feature_points_agm.pkl')
model = TripletLossNet(device)
model.eval()


@app.route(url_prefix + '/')
def hello_world():
    return 'hello world'

@app.route(url_prefix + '/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route(url_prefix + '/train')
def train():
    return 'Spend Time:\t'


@app.route(url_prefix + '/test', methods=['post'])
def test():
    image_url = request.form.get("image_url")
#     image_url = request.headers.get("image_url")
#     for item in request.headers:
#         print(item)
    try:
        k = int(request.form.get("k"))
        k = k if k is not None and k >= 1 else 1
    except:
        k = 3
    print(image_url)
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if image_url and allowed_file(image_url):
        fname = secure_filename(image_url)
        ext = fname.rsplit('.', 1)[1]
        new_filename = str(uuid.uuid3(uuid.NAMESPACE_URL, fname)) + '.' + ext
        image_path = os.path.join(file_dir, new_filename)
        urllib.request.urlretrieve(image_url, filename=image_path)

        image = Image.open(image_path)
        res = detector.detect_image(image)
        out_file_name = os.path.join(file_dir, 'yolo', new_filename)
        res.save(out_file_name)
        print(out_file_name)
        predict = model.predict_top(out_file_name, k)
        skus = predict['sku']
        sku_url = []
        for sku in skus:
            sku_url.append(SKU_URL_PREFIX + sku + '.jpg')
        predict['sku_url'] = sku_url
        out_url = root_url + '/' + app.config['UPLOAD_FOLDER'] + '/yolo/' + new_filename
        return jsonify({"success": 0, "msg": "上传成功", "predict": predict, "url":out_url})

    else:
        return jsonify({"success": 1001, "msg": "上传失败"})

@app.route(url_prefix + '/test_multi', methods=['post'])
def test_multi():
    image_url = request.form.get("image_url")
#     image_url = request.headers.get("image_url")
#     for item in request.headers:
#         print(item)
    try:
        k = int(request.form.get("k"))
        k = k if k is not None and k >= 1 else 1
    except:
        k = 3
    print(image_url)
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if image_url and allowed_file(image_url):
        fname = secure_filename(image_url)
        ext = fname.rsplit('.', 1)[1]
        new_filename = str(uuid.uuid3(uuid.NAMESPACE_URL, fname)) + '.' + ext
        image_path = os.path.join(file_dir, new_filename)
        urllib.request.urlretrieve(image_url, filename=image_path)

        image = Image.open(image_path)
        boxed_list = detector.detect_image_multi(image)
        out_file_name_list = []
        out_url_list = []
        for i in range(len(boxed_list[0])):
            out_file_name = os.path.join(file_dir, 'yolo', '%02d_' % i + new_filename)
            boxed_list[0][i].save(out_file_name)
            out_file_name_list.append(out_file_name)
            out_url_list.append(root_url + '/' + app.config['UPLOAD_FOLDER'] + '/yolo/' + '%02d_' % i + new_filename)
        print(out_file_name_list)
        predict = []
        if len(out_file_name_list) > 0:
            predict = model.predict_batch(out_file_name_list, k)
        elif len(boxed_list[1]) == 0:
            predict = model.predict_batch([image_path], k)
        for pr in predict:
            skus = pr['sku']
            sku_url = []
            for sku in skus:
                sku_url.append(SKU_URL_PREFIX + sku + '.jpg')
            pr['sku_url'] = sku_url
        
       
        for i in range(len(boxed_list[1])):
            out_file_name = os.path.join(file_dir, 'yolo', 'other_%02d_' % i + new_filename)
            boxed_list[1][i].save(out_file_name)
            out_file_name_list.append(out_file_name)
            out_url_list.append(root_url + '/' + app.config['UPLOAD_FOLDER'] + '/yolo/' + 'other_%02d_' % i + new_filename)
        
        predict += [{'name': [], 'prob': [], 'sku': []} for _ in range(len(boxed_list[1]))]
          
        
        return jsonify({"success": 0, "msg": "上传成功", "predict": predict, "out_url_list":out_url_list})

    else:
        return jsonify({"success": 1001, "msg": "上传失败"})    
    
    
@app.route(url_prefix + '/retrain', methods=['post'])
def retrain():
    image_url = request.form.get("image_url")
    name = request.form.get("name")
#     image_url = request.headers.get("image_url")
#     for item in request.headers:
#         print(item)
    print(image_url)
#     print(name.encode('utf-8').decode('ascii'))
    image_urls = image_url.split(',')
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        
    urls = []   
    face_path = ''
    class_uuid = str(uuid.uuid3(uuid.NAMESPACE_URL, name))
    for image_url in image_urls:
        
        if image_url and allowed_file(image_url):
            fname = secure_filename(image_url)
            ext = fname.rsplit('.', 1)[1]
            new_filename = str(uuid.uuid3(uuid.NAMESPACE_URL, fname)) + '.' + ext
            image_path = os.path.join(file_dir, new_filename)
            urllib.request.urlretrieve(image_url, filename=image_path)
            face_path = image_path


            image = Image.open(image_path)
            res = detector.detect_image(image)
            out_file_name = os.path.join(file_dir, 'yolo', new_filename)
            res.save(out_file_name)
      
            urls.append(out_file_name)
        else:
            return jsonify({"success": 1001, "msg": "上传失败"})
    print(urls)
#     new_class_repre = model.retrain(urls, name, class_uuid)

    info = model.retrain(urls, name, class_uuid)
    if info['msg'] == 'success':
        shutil.copy(face_path , os.path.join(basedir, 'static/sku_face')+ '/' + class_uuid + '.jpg')
        return jsonify({"success": 0, "msg": "模型训练成功！"})
    
    else:
        return jsonify({"success": 1, "msg": "模型训练失败！", 'name': info['similar_object_name'], 'url': SKU_URL_PREFIX + info['similar_object_sku'] + '.jpg'})
#     print(new_class_repre)
#     return jsonify({"success": 0, "msg": "模型训练成功！"})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) # 443, ssl_context ='adhoc'
