import uuid

from flask import Flask, render_template, redirect, url_for, session, request
from PIL import Image

from localization import is_defective, save_activation_map

app = Flask(__name__)
app.config.from_object(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if not request.files['raw_image']:
        return 'No file uploaded'
    else:
        image = request.files['raw_image']
        # Save the image to the tmp folder
        image_path = f"tmp/{str(uuid.uuid4().hex)}.jpeg"
        image.save(image_path)
        # Check if the image is defective
        defective = is_defective(image_path)
        # Save the activation map
        save_activation_map(image_path)
        if defective:
            print('Defective')
            return 'Defective'
        else:
            print('Not Defective')
            return 'Not Defective'
    