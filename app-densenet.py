
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
from densenet.densenet import Densenet
# from resnet.resnet import resnet
from preprocess import preprocess
import uuid
import urllib.request
from werkzeug.utils import secure_filename
# import torch

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
# device = torch.device("cuda: 3" if torch.cuda.is_available() else "cpu")
print('basedir:', basedir)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



# model = resnet(device)
model = Densenet()



@app.route('/infer-ed58ae0d-a251-42e0-aa9a-329181df32d6')
def hello_world():
    return 'hello world'


@app.route('/train')
def train():
    return 'Spend Time:\t'


@app.route('/infer-ed58ae0d-a251-42e0-aa9a-329181df32d6/test', methods=['post'])
def test():
    image_url = request.form.get("image_url")
#     image_url = request.headers.get("image_url")
#     for item in request.headers:
#         print(item)
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
        # image_url.save(os.path.join(file_dir, new_filename))
        image = preprocess(image_path)
        predict = model.predict(image)
        return jsonify({"success": 0, "msg": "上传成功", "predict": predict})

    else:
        return jsonify({"error": 1001, "msg": "上传失败"})



# # 上传文件
# @app.route('/up_photo', methods=['POST'], strict_slashes=False)
# def api_upload():
#     file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
#     if not os.path.exists(file_dir):
#         os.makedirs(file_dir)
#     f = request.files['photo']
#     if f and allowed_file(f.filename):
#         fname = secure_filename(f.filename)
#         ext = fname.rsplit('.', 1)[1]
#         new_filename = str(uuid.uuid3(uuid.NAMESPACE_URL, fname)) + '.' + ext
#         f.save(os.path.join(file_dir, new_filename))
#         return jsonify({"success": 0, "msg": "上传成功", "path": os.path.join(app.config['UPLOAD_FOLDER'], new_filename)})
#     else:
#         return jsonify({"error": 1001, "msg": "上传失败"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
