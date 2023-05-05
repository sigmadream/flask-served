import argparse
import io

import torch
from flask import Flask, request
from PIL import Image

app = Flask(__name__)
models = {}

DETECTION_URL = '/api/predict'


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return

    if request.files.get('image'):
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        results = model(im, size=640)
        return results.pandas().xyxy[0].to_json(orient='records')


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', force_reload=True, skip_validation=True)
    app.run(host='0.0.0.0', debug=True, port=5000)
