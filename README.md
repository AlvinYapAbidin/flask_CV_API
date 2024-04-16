# Flask CV API
This repository contains a simple Flask API designed for image segmentation utilizing advanced machine learning models such as SAM (Segment Anything Model) and FastSAM.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
Prerequisites

Ensure you have Python installed on your machine, along with Flask and other required libraries:

    pip install flask numpy opencv-python-headless pillow

## Installation

### Clone the repository:

    git clone https://github.com/yourusername/flask_CV_API.git
    cd flask_CV_API

### Download the Model Weights:

For FastSAM model:

    wget -O FastSAM.pt https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt

For SAM model:

    wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Note: The sam_vit_h_4b8939.pth file is over 2.5GB, ensure you have sufficient storage available.

### Setup the Flask application:
Place the downloaded .pt files in a directory accessible to your application, or update the paths in the create_mask_FastSAM and create_mask_SAM functions respectively.

## Running the API

To start the server, run:

python app.py

This will start the Flask server on http://localhost:5000. You can make POST requests to /segment to process your images using **Postman** (https://www.postman.com/). VSCode has an extension for Postman as well

## Using the API

To use the API:

1. Send a POST request to /segment with an image file and the coordinates x and y marking the point of interest for segmentation.
2. The API will return the path to the segmented image saved on the server.
