import os
import sys
import json
import base64
import io
import requests
from flask import Flask, request
from PIL import Image

from urllib.parse import unquote

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def webhook():

    payload = request.json
    # print(payload["photo"])
    im_string = payload["photo"]
    im = base64.b64decode(im_string)
    img = Image.open(io.BytesIO(im))
    img.show()  # showing sended photo
    # print(type(im))

    return "200"


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 5000))
