import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import urllib
import uuid

from main import infer_set
from Model import Model, DecoderType

UPLOAD_FOLDER = os.path.join(os.path.abspath(
                os.path.join(os.path.abspath(__file__), '..', '..')), 'FlaskData')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnSummary = '../model/summary.json'
    fnInfer = '../data/test.png'
    fnCorpus = '../data/corpus.txt'


def get_names():
    with open('../data/BestDemo.txt', 'r') as f:
        names = f.read()
    return names.splitlines()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.values)
        print(request.values['dataUrl'])
        data = request.values
        response = urllib.request.urlopen(data['dataUrl'])
        unique_filename = str(uuid.uuid4())
        filename = f'{unique_filename}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(response.file.read())
        try:
            recognized, probability = infer_set(app.model, filepath, get_names())
            returned = {'recognized': str(recognized[0]), 'probability': str(probability[0][0])}
        except IndexError:
            returned = {'error': 'no name found'}
        os.remove(filepath)
        return jsonify(returned)


if __name__ == '__main__':
    model = Model(open(FilePaths.fnCharList).read(), DecoderType.WordBeamSearch, mustRestore=True)
    app.model = model
    app.run()