import json
import os
import sys
import time
import torch
from flask import Flask, request
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
sys.path.append("./src")

# pylint: disable=wrong-import-position

from datasets import LandmarkHDF5Dataset
from utils import ServerUtils,TrainTransforms
import utils.config as cfg
# pylint: enable=wrong-import-position

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODEL_PATH = "data/results/golden-laughter-3/model/metric-golden-laughter-3-e9.path"
MODEL_DEVICE = "cuda"
IMAGE_KEY = "photo"
NUM_RES_KEY = "numResults"

app = Flask(__name__)

server_util = ServerUtils()


tf_list = TrainTransforms(mean=MEAN, std=STD, as_hdf5=True)

train_dataset = LandmarkHDF5Dataset(
    root=cfg.HDF5_DIR_PATH,
    running_mode="train",
    transforms_list=tf_list,
    measure_time=True,
    is_metric=True,
)

model = server_util.load_model(MODEL_PATH,device=MODEL_DEVICE)
print(cfg.TERMINAL_INFO,"model loaded")

match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
inference_model = InferenceModel(model, match_finder=match_finder)
inference_model.train_knn(train_dataset)
print(cfg.TERMINAL_INFO, "train knn done")






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



        pil_request_image = server_util.image_base64_to_pil(img_string)
        # image_from_req.show(title="Image from request")

        start_time = time.time()

        tensor_request = server_util.image_pil_to_tensor(pil_request_image)
        tensor_image = torch.unsqueeze(tensor_request, 0)
        distances, indices = inference_model.get_nearest_neighbors(tensor_image.float(), k=num_results)

        nearest_imgs = [server_util.image_pil_to_tensor(train_dataset[i][0]) for i in indices.cpu()[0]]

        # server_util.imshow(nearest_imgs)

        np_response_images = [server_util.image_tensor_to_numpy(im) for im in nearest_imgs]
        pil_response_images = [server_util.image_numpy_to_pil(im) for im in np_response_images]
        server_util.combine_images(pil_response_images,show_image=True) # shows randomly selected images
        bytes_response_images = [server_util.image_pil_to_base64(image) for image in pil_response_images]
        list_of_nested_dics =[{"image":image} for image in bytes_response_images]


        response_message = {
                            "hello":"world",
                            "images": list_of_nested_dics
                        }
        response_status = 200
        end_time = time.time()
        print(cfg.TERMINAL_INFO,f"TIME : {(end_time-start_time):.2f}")
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



    
    # app.debug = True # debuging mode needs to be changed later

    app.run(host="0.0.0.0", port=os.environ.get("PORT", 5000))
    