
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort, Blueprint
# from densenet.densenet import Densenet
from preprocess import preprocess
import uuid
import urllib.request
from werkzeug.utils import secure_filename
from yolo.persondet import PersonDetector
import cv2
import base64

url_prefix = '/infer-ed58ae0d-a251-42e0-aa9a-329181df32d6'
root_url = 'http://106.75.34.228:82' + url_prefix
# url_prefix = ''
# root_url = 'http://127.0.0.1:5000'

# bp = Blueprint('burritos', __name__,
#                template_folder='templates')
app = Flask(__name__)
# app.register_blueprint(bp, url_prefix=url_prefix)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

basedir = os.path.abspath(os.path.dirname(__file__))
print('basedir:', basedir)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# densenet = Densenet()
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
VOC_CLASSES = ('person')
FONT = cv2.FONT_HERSHEY_SIMPLEX
detector = PersonDetector()


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
        # predict = densenet.predict(image)
        predict = 'not for now'
        return jsonify({"success": 0, "msg": "上传成功", "predict": predict})

    else:
        return jsonify({"error": 1001, "msg": "上传失败"})


@app.route(url_prefix + '/yolo', methods=['post'])
def yolo():
    image_url = request.form.get("image_url")
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(os.path.join(file_dir, 'yolo')):
        os.makedirs(os.path.join(file_dir, 'yolo'))
    if image_url and allowed_file(image_url):
        fname = secure_filename(image_url)
        ext = fname.rsplit('.', 1)[1]
        new_filename = str(uuid.uuid3(uuid.NAMESPACE_URL, fname)) + '.' + ext
        image_path = os.path.join(file_dir, new_filename)
        urllib.request.urlretrieve(image_url, filename=image_path)

        image = cv2.imread(image_path)
        _labels, _scores, _coords = detector.predict(image)
        for labels, scores, coords in zip(_labels, _scores, _coords):
            cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3],
                          2)
            cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores),
                        (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
        # cv2.putText(image,'len:%d'%len(_labels),(100,100),FONT,5, COLORS[labels % 3], 2)
        out_file_name = os.path.join(file_dir, 'yolo', new_filename)
        cv2.imwrite(out_file_name, image)
        print(out_file_name)
        with open(out_file_name,'rb') as f:
            base64_data = base64.b64encode(f.read())
        out_url = root_url + '/' + app.config['UPLOAD_FOLDER'] + '/yolo/' + new_filename
        return jsonify({"success": 0, "msg": "上传成功", "base64_data": str(base64_data)[2:-1], "img_url":out_url})

    else:
        return jsonify({"error": 1001, "msg": "上传失败"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

