from flask import Flask, render_template, redirect, url_for, session, request

app = Flask(__name__)
app.config.from_object(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['raw_image']
    # Process image using model and return result
    return 'Successfully uploaded image'
    