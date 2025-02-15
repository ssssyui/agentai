from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import requests
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')


@app.route('/ocr', methods=['POST'])
def ocr_image():
    image_url = request.json.get('url')
    print("111"+str(image_url))
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    # Fetch the image from the URL, ignoring SSL verification (use with caution)
    try:
        response = requests.get(image_url, verify=False)
        response.raise_for_status()  # Raises HTTPError for bad responses
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 400

    try:
        image = Image.open(BytesIO(response.content))
        image_np = np.array(image)

        # Perform OCR
        result = ocr.ocr(image_np, cls=True)

        # Convert result to JSON
        json_output = []
        for line in result:
            for word_info in line:
                word = {
                    "text": word_info[1][0],
                    "confidence": word_info[1][1],
                    "position": [
                        {"x": int(word_info[0][0][0]), "y": int(word_info[0][0][1])},
                        {"x": int(word_info[0][1][0]), "y": int(word_info[0][1][1])},
                        {"x": int(word_info[0][2][0]), "y": int(word_info[0][2][1])},
                        {"x": int(word_info[0][3][0]), "y": int(word_info[0][3][1])}
                    ]
                }
                json_output.append(word)

        return jsonify(json_output), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)