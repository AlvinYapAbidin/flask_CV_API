from flask import Flask, request, send_file
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the image processing server!"

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image provided", 400
    
    image_file = request.files['image']
    x = int(request.form.get('x',0))
    y = int(request.form.get('y',0))

    image = Image.open(image_file.stream)
    image_np = np.array(image)

    processed_image_np = cv2.rectangle(image_np,x,y,Scalar(0,0,255),3,0)
    process_image = Image.fromarray(processed_image_np)

    byte_io =  io.BytesIO()
    process_image.save(byte_io, 'PNG')
    byte_io.seek(0)

    return send_file(byte_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)