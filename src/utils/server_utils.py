import base64
import json
import io
import os
import random
from PIL import Image
import numpy as np

import utils.config as cfg


class ServerUtils:
    """
    Helper class for server
    """

    def __init__(self) -> None:
        pass

    def __image_path_to_base64(self, image_str: str) -> str:
        """
        Converts given image path to base64
        """
        with open(image_str, "rb") as image:
            return base64.b64encode(image.read()).decode("utf-8")

    def image_pil_to_base64(self, image: Image.Image) -> str:
        """
        Converts a PIL Image to base64 string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=image.format)
        buffer = buffer.getvalue()
        return base64.b64encode(buffer).decode("utf-8")

    def image_base64_to_pil(self, img_string: str) -> Image.Image:
        """
        convert string in base64 format to PIL image
        """
        img_bytes = base64.b64decode(img_string)
        img_pil = Image.open(io.BytesIO(img_bytes))
        return img_pil

    def get_json_response(self, image_path: str) -> str:
        """
        Returns JSON message with encoded image
        """
        encoded = self.__image_path_to_base64(image_path)

        message = {"returned_image": encoded}
        return json.dumps(message)

    def get_json_response_from_pil(self, image: Image) -> str:
        """
        Returns JSON message with base64 encoded image from PIL Image
        """
        encoded = self.image_pil_to_base64(image)

        message = {"returned_image": encoded}
        return json.dumps(message)

    def combine_images(self, images: list[Image.Image], show_image: bool = False) -> Image.Image:
        """
        (for local tests only) stack images to one horizontaly
        """
        resized_images_list = list(map(lambda photo: photo.resize((300, 500)), images))
        np_images_list = [np.array(image) for image in resized_images_list]
        stacked_image = np.hstack(np_images_list)
        image = Image.fromarray(stacked_image)
        if show_image:
            image.show()

        return image

    def __get_random_images_paths(self, num_of_images: int) -> list[str]:
        """
        return paths to random images from landmark dataset img dir
        """
        dir_path = os.path.join(cfg.LANDMARK_DATASET_PATH, cfg.LM_IMGS_DIR_PATH)
        images = random.choices(os.listdir(dir_path), k=num_of_images)
        images_paths = [os.path.join(dir_path, im) for im in images]

        return images_paths

    def get_random_pil_images(self, num_of_images: int) -> list[Image.Image]:

        """
        return list of random PIL images  from dataset
        """
        paths = self.__get_random_images_paths(num_of_images)

        return [Image.open(path) for path in paths]
