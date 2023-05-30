import os
import threading
import time
import uuid
from io import BytesIO
import base64

from flask import Flask, render_template, request, jsonify
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

@app.route('/process-image', methods=['POST'])
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
        
        # Check if the image is defective + delete the image after 15 seconds
        threading.Thread(target=garbage_collection, args=(image_path, 15), daemon=True).start()
        
        defective, confidence = is_defective(image_path)
        
        if defective:
            save_activation_map(image_path)
            # Plot and save the output image to the tmp folder + delete the image after 60 seconds
            threading.Thread(target=garbage_collection, args=(output_image_path, 60), daemon=True).start()
            
            with Image.open(output_image_path) as img:
                image_data = BytesIO()
                img.save(image_data, format='PNG')
                image_data.seek(0)

                # Encode the image data as base64
                base64_image = base64.b64encode(image_data.getvalue()).decode()
                
            response = {'image': base64_image,
                        'is_defective': defective,
                        'confidence': confidence}

            return jsonify(response)
        else:
            response = {'is_defective': defective,
                        'confidence': confidence}
            
            return jsonify(response)
    