from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['jpg', 'bmp'])
model = load_model('models/model1.h5')
model_f = load_model('models/model2.h5')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Prédiction par genre
def predict(file):
    classes = ['Male', 'Female']
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_array, (28, 28))
    img_resize = np.array(img_resize).reshape(-1, 28, 28, 1)
    img_resize = img_resize.astype('float32')
    img_resize = img_resize / 255.0
    result = model.predict(img_resize)
    dict2 = {}
    for i in range(2):
        dict2[result[0][i]] = classes[i]
    res = result[0]
    res.sort()
    res = res[::-1]
    results = res[:3]
    answer = {}
    for i in range(2):
        answer[dict2[results[i]]] = (results[i] * 100).round(2)
    print(answer)
    return answer

# Prédiction par type de doigt
def predict_finger(file):
    classes = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_array, (28, 28))
    img_resize = np.array(img_resize).reshape(-1, 28, 28, 1)
    img_resize = img_resize.astype('float32')
    img_resize = img_resize / 255.0
    result = model_f.predict(img_resize)
    dict2 = {}
    for i in range(5):
        dict2[result[0][i]] = classes[i]
    res = result[0]
    res.sort()
    res = res[::-1]
    results = res[:3]
    answer = {}
    for i in range(3):
        answer[dict2[results[i]]] = (results[i] * 100).round(2)
    print(answer)
    return answer


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('Aucun fichier')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Aucun image sélectionné pour uploader')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        answer = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        f_answer = predict_finger(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        for index, value in answer.items():
            flash(f'Cette empreinte de {index}: {value} %   ')
        for index, value in f_answer.items():
            flash(f'Avec type du doigt {index}: {value} %   ')
        return render_template('index.html', filename=filename)
    else:
        flash('Images allowés : .jpg, .bmp')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()
