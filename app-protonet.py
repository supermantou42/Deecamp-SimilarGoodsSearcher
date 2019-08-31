from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort, Blueprint
from preprocess import preprocess
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import uuid
import urllib.request
from werkzeug.utils import secure_filename

# from densenet.densenet import Densenet

from ProtoNet.protonet import ProtoNet
from yolo_tf.yolo import YOLO
from PIL import Image

from yolo_pt.persondet import PersonDetector
import cv2
# import base64

url_prefix = '/infer-ed58ae0d-a251-42e0-aa9a-329181df32d6'
root_url = 'http://106.75.34.228:82' + url_prefix
# url_prefix = '/prefix'
# root_url = 'http://127.0.0.1:5000'

app = Flask(__name__)
# bp = Blueprint('burritos', __name__)
# app.register_blueprint(bp, url_prefix=url_prefix)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

basedir = os.path.abspath(os.path.dirname(__file__))
print('basedir:', basedir)
file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
print('filedir:', file_dir)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# model = Densenet()
model = ProtoNet()
# COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# VOC_CLASSES = ('person')
# FONT = cv2.FONT_HERSHEY_SIMPLEX
# detector = PersonDetector()
detector = YOLO()


@app.route(url_prefix + '/')
def hello_world():
    return 'hello world'


@app.route(url_prefix + '/train')
def train():
    return 'Spend Time:\t'


@app.route(url_prefix + '/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)



@app.route(url_prefix + '/test', methods=['post'])
def test():
    image_url = request.form.get("image_url")
    try:
        k = int(request.form.get("k"))
        k = k if k is not None and k >= 1 else 1
    except:
        k = 1
    if image_url and allowed_file(image_url):
        fname = secure_filename(image_url)
        ext = fname.rsplit('.', 1)[1]
        new_filename = str(uuid.uuid3(uuid.NAMESPACE_URL, fname)) + '.' + ext
        image_path = os.path.join(file_dir, new_filename)
        try:
            urllib.request.urlretrieve(image_url, filename=image_path)
        except:
            return jsonify({"error": 1001, "msg": "图片加载失败"})      
        image = Image.open(image_path)
        res = detector.detect_image(image)
        out_file_name = os.path.join(file_dir, 'yolo', new_filename)
        res.save(out_file_name)
#         out_file_name = image_path
    
    
    
        # image_url.save(os.path.join(file_dir, new_filename))
        image = preprocess(out_file_name, 84)
        # predict = densenet.predict(image)
        name, dist = model.find(image, k)
        predict = {"name":name,"prob":dist}
        # predict = 'not for now'
        return jsonify({"success": 0, "msg": "上传成功",
                        "predict": predict})

    else:
        return jsonify({"error": 1001, "msg": "上传失败"})

    
# @app.route(url_prefix + '/yolo', methods=['post'])
# def yolo():
#     image_url = request.form.get("image_url")
# #     file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
#     if not os.path.exists(os.path.join(file_dir, 'yolo')):
#         os.makedirs(os.path.join(file_dir, 'yolo'))
#     if image_url and allowed_file(image_url):
#         fname = secure_filename(image_url)
#         ext = fname.rsplit('.', 1)[1]
#         new_filename = str(uuid.uuid3(uuid.NAMESPACE_URL, fname)) + '.' + ext
#         image_path = os.path.join(file_dir, new_filename)
#         try:
#             urllib.request.urlretrieve(image_url, filename=image_path)
#         except:
#             return jsonify({"error": 1001, "msg": "图片加载失败"})
# #         image = Image.open(image_path)
# #         res = detector.detect_image(image)
# #         out_file_name = os.path.join(file_dir, 'yolo', new_filename)
# #         res.save(out_file_name)
# #         print(out_file_name)
# #         with open(out_file_name,'rb') as f:
# #             base64_data = base64.b64encode(f.read())
        
#         image = cv2.imread(image_path)
#         _labels, _scores, _coords = detector.predict(image)
  
#         if len(_labels) > 0:
#             for labels, scores, coords in zip(_labels, _scores, _coords):
#                 cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
#                 cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores),
#                             (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
#         out_file_name = os.path.join(file_dir, 'yolo', new_filename)
#         cv2.imwrite(out_file_name, image)
#         out_url = root_url + '/' + app.config['UPLOAD_FOLDER'] + '/yolo/' + new_filename
#         return jsonify({"success": 0, "msg": "上传成功", "img_url":out_url, "cnt":len(_labels)}) # "base64_data": str(base64_data)[2:-1],

#     else:
#         return jsonify({"error": 1001, "msg": "上传失败"})
    
@app.route(url_prefix + '/add_sample', methods=['post'])
def add_sample():
    name = request.form.get("name")
    image_url_raw = request.form.get("image_url")
    if not image_url_raw or not name or len(name) == 0:
        return jsonify({"error": 1001, "msg": "上传失败, 无数据"})
    image_url_list = [i for i in image_url_raw.split(',') if i != '']
    for url in image_url_list:
        if not allowed_file(url):
            return jsonify({"error": 1001, "msg": "上传失败, 格式不正确，错误的url为 %s" % url})

    image_list = []
    for url in image_url_list:
        fname = secure_filename(url)
        ext = fname.rsplit('.', 1)[1]
        new_filename = str(uuid.uuid3(uuid.NAMESPACE_URL, fname)) + '.' + ext
        image_path = os.path.join(file_dir, new_filename)
        try:
            urllib.request.urlretrieve(image_url, filename=image_path)
        except:
            return jsonify({"error": 1001, "msg": "图片加载失败"})

        image = preprocess(image_path, 84)
        image_list.append(image)

    model.add_sample(image_list, name)
    # predict = 'not for now'
    return jsonify({"success": 0, "msg": "添加成功"})

if __name__ == '__main__':
    # app.config['APPLICATION_ROOT'] = '/prefix'
    app.run(host='0.0.0.0', port=8080)
