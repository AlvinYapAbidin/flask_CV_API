from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MASK_FOLDER'] = 'static/masks/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MASK_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return "Image processing server"

@app.route('/segment', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify(result=-1, message="No image provided"), 400
    
    image_file = request.files['image']
    if not allowed_file(image_file.filename):
        return jsonify(result=-1, message="File format not supported"), 400
    
    # save original image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    try: # 'try-except'to catch and handle exceptions returning server error (500)
        image = Image.open(image_path)
        image_np = np.array(image)

        height, width = image_np.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        x = int(request.form.get('x', 0))
        y = int(request.form.get('y',0))
        cv2.rectangle(mask,(x,y), (x+100, y+100), (255), thickness=-1)
        mask_image = Image.fromarray(mask)

        # save mask
        mask_filename = f"mask_{unique_filename}"
        mask_path = os.path.join(app.config['MASK_FOLDER'], f"mask_{image_file.filename}")
        mask_image.save(mask_path)
    except Exception as e:
        return jsonify(result=-1, message=str(e)), 500

    return jsonify(result=0, mask=mask_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)

# Testing: curl -X POST -F 'image=@image/lena.png' -F 'x=50' -F 'y=50' http://127.0.0.1:5000/segment
