import json
import os
import sys

from flask import Flask, request

sys.path.append("./src")

# pylint: disable=wrong-import-position
from utils.server_utils import ServerUtils
# pylint: enable=wrong-import-position



IMAGE_KEY = "photo"
NUM_RES_KEY = "numResults"


app = Flask(__name__)

server_util = ServerUtils()

@app.route("/", methods=["GET"])
def webhook():
    """
    basic webhoor for GET requesst
    """
    return "200"

@app.route("/", methods=["POST"])


def receive_post_image():
    """
    receive by POST request image in json format
    """

    try:

        payload = request.json
        img_string = payload[IMAGE_KEY]
        num_results = payload[NUM_RES_KEY]


        #FOR TESTING PURPOUSE
        image_from_req = server_util.image_base64_to_pil(img_string)
        image_from_req.show(title="Image from request")

        pil_response_images = server_util.get_random_pil_images(num_results)

        #FOR TESTING PURPOUSE
        server_util.combine_images(pil_response_images,show_image=True) # shows randomly selected images



        bytes_response_images = [server_util.image_pil_to_base64(image) for image in pil_response_images]

        list_of_nested_dics =[{"image":image} for image in bytes_response_images]
        response_message = {
                            "hello":"world",
                            "images": list_of_nested_dics
                        }
        print(type(num_results),num_results)




        response_status = 200
    except TypeError:
        if request.is_json:
            payload = request.json
            try:
                img_value = payload[IMAGE_KEY]
            except TypeError:
                img_value = None
        response_message = {"is_json": request.is_json,
                             "img_value_type": str(type(img_value)),
                             "img_value": str(img_value)}

        response_status = 400

    return app.response_class(
        response=json.dumps(response_message), status=response_status, mimetype="application/json")


if __name__ == "__main__":
    app.debug = True # debuging mode needs to be changed later

    app.run(host="0.0.0.0", port=os.environ.get("PORT", 5000))
    