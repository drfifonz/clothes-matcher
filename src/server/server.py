import os
import sys
import json
import base64
import io
import requests
from flask import Flask, request, jsonify
from PIL import Image

from urllib.parse import unquote


IMAGE_KEY = "photo"

app = Flask(__name__)


@app.route("/", methods=["GET"])
def webhook():
    return "200"

@app.route("/", methods=["POST"])
def recive_post_image():
    try:
        
        payload = request.json
        img_string = payload[IMAGE_KEY]
        img_bytes = base64.b64decode(img_string)
        img_pil = Image.open(io.BytesIO(img_bytes))
        # img_pil.show()  # showing sended photo
        
        response_messages = {"response_message": "OK"}
        response_status = 200
    except: 
        if request.is_json:
            payload = request.json
            try:
                 img_value = payload[IMAGE_KEY]
            except:
                img_value  = None

        response_messages = {"is_json": request.is_json,
                             "img_value_type": str(type(img_value)),
                             "img_value": str(img_value)}
        response_status = 406

    return app.response_class(response=json.dumps(response_messages), status=response_status, mimetype="application/json")




if __name__ == "__main__":
    app.debug = True # debuging mode needs to be changed later

    app.run(host="0.0.0.0", port=os.environ.get("PORT", 5000))
    