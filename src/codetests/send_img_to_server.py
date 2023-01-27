import sys
import base64
import io
import json

import requests
from PIL import Image

sys.path.append("./src")

# pylint: disable=wrong-import-position
from utils.server_utils import ServerUtils

# pylint: enable=wrong-import-position

URL = "http://localhost:5000/"

NUM_RESULTS = 10

server_utils = ServerUtils()

with open("data/temp/tshirt_200.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read())
    # print(type(encoded_image))
    # print(*encoded_image)
    decoded_image = base64.b64decode(encoded_image)

    im = Image.open(io.BytesIO(decoded_image))
    # im.show()


headers = {"Content-Type": "application/json"}
data = {
    "numResults": NUM_RESULTS,
    "photo": encoded_image.decode(),
}

# im_list = [im, im, im]
# new_image = server_utils.combine_images(im_list, show_image=False)
# new_image.show()

# lista = server_utils.get_random_pil_images(5)
# new_image = server_utils.combine_images(lista, show_image=True)

req = requests.post(url=URL, data=json.dumps(data), headers=headers, timeout=10000)
# resp_message = req.content.decode("utf-8")
# res = json.loads(resp_message)
# print(json.dumps(res, indent=4), req.status_code)
print(req.status_code)
