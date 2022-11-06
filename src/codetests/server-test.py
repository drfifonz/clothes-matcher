import requests
import json
import base64
import io
from PIL import Image

URL = "http://localhost:5000/"


with open("data/image.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read())
    print(type(encoded_image))
    # print(*encoded_image)
    decoded_image = base64.b64decode(encoded_image)

    im = Image.open(io.BytesIO(decoded_image))
    # im.show()

headers = {"Content-Type": "application/json"}
data = {"hello": "world", "photo": encoded_image.decode()}

requests.post(url=URL, data=json.dumps(data), headers=headers)
# requests.post(url=URL, json=jsondata, headers=headers)
