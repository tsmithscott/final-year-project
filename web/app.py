import os
import threading
import time
import uuid

from flask import Flask, redirect, render_template, request, session, url_for
from localization import is_defective, save_activation_map
from PIL import Image

app = Flask(__name__)
app.config.from_object(__name__)


def garbage_collection(path: str, time_seconds: int):
    time.sleep(time_seconds)
    if os.path.exists(path):
        os.remove(path)
    exit(0)
        

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if not request.files['raw_image']:
        return 'No file uploaded'
    else:
        # Get the image from the request
        image = request.files['raw_image']
        
        # Save the image to the tmp folder
        file_name = str(uuid.uuid4().hex)
        image_path = f"tmp/{file_name}.jpeg"
        output_image_path = os.path.join('tmp', f'{file_name}-output.png')
        image.save(image_path)
        
        # Check if the image is defective + delete the image after 30 seconds
        defective = is_defective(image_path)
        threading.Thread(target=garbage_collection, args=(image_path, 30), daemon=True).start()

        # Plot and save the output image to the tmp folder + delete the image after 30 seconds
        save_activation_map(image_path)
        threading.Thread(target=garbage_collection, args=(output_image_path, 30), daemon=True).start()
        
        if defective:
            print('Defective')
            return 'Defective'
        else:
            print('Not Defective')
            return 'Not Defective'
    