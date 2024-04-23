# Setup
from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import os
from segment_anything import SamPredictor, sam_model_registry
# from fastsam import FastSAM, FastSAMPrompt # requires the fastsam folder in order to import
import pkg_resources

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MASK_FOLDER'] = 'static/masks/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MASK_FOLDER'], exist_ok=True)

### Routing ###


# @app.route('/')
# def index():
#     return "Image processing server"

@app.route('/', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify(result=-1, message="No image provided"), 400
    
    image_file = request.files['image']
    if not allowed_file(image_file.filename):
        return jsonify(result=-1, message="File format not supported"), 400
    
    # Save original image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)
    
    x = int(request.form.get('x', 0))
    y = int(request.form.get('y', 0))

    try:
        mask_path = create_mask_SAM(image_path, x, y, image_file.filename)
    except Exception as e:
        return jsonify(result=-1, message=str(e)), 500

    return jsonify(result=0, mask=mask_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# def create_mask_FastSAM(image_path, x, y, filename):
#     try:
#         model = FastSAM('FastSAM.pt')
#         IMAGE_PATH = image_path
#         DEVICE = 'cpu'
#         everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
#         prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

#         ann = prompt_process.point_prompt(points=[[x, y]], pointlabel=[1])
        
#         if ann.ndim == 3 and ann.shape[0] == 1:
#             ann = np.squeeze(ann, axis=0)  # converting the array from (1, height, width) to (height, width),
        
#         mask = ann
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8))

#         mask_path = os.path.join(app.config['MASK_FOLDER'], f"mask_{filename}")
#         # prompt_process.plot(annotations=ann, output_path=mask_path)
#         mask_image.save(mask_path)

#         return  mask_path
#     except Exception as e:
#         raise Exception(f"Failed to process image:{str(e)}")

def create_mask_SAM(image_path, x, y, filename):
    try:
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        predictor = SamPredictor(sam)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_point = np.array([[x, y]]) # Truck example (500,375)
        input_label = np.array([1]) 

        # Prediction
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # Highest scoring mask
        best_mask_index = np.argmax(scores)
        best_mask = masks[best_mask_index]

        # Mask to a PIL Image conversion + save
        mask_image = Image.fromarray((best_mask * 255).astype(np.uint8))
        mask_path = os.path.join(app.config['MASK_FOLDER'], f"mask_{filename}")
        mask_image.save(mask_path)

        return mask_path
    except Exception as e:
        raise Exception(f"Failed to process image: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
