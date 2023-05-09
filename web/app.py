import uuid

from flask import Flask, render_template, redirect, url_for, session, request
from PIL import Image

from localization import load_trained_model, is_defective, save_activation_map

app = Flask(__name__)
app.config.from_object(__name__)
model = load_trained_model()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if not request.files['raw_image']:
        return 'No file uploaded'
    else:
        image = request.files['raw_image']
        image_path = f"tmp/{str(uuid.uuid4().hex)}.jpeg"
        image.save(image_path)
        if is_defective(image_path) == True:
            print('Defective')
            return 'Defective'
        else:
            print('Not Defective')
            return 'Not Defective'
    