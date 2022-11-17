import base64
import json
import io
from PIL import Image


class ServerUtil():
    """
    Helper class for creating JSON responses
    """
    def __init__(self) -> None:
        pass

    def __image_path_to_base64(self, image_str: str) -> str:
        """
        Converts given image path to base64
        """
        with open(image_str, "rb") as image:
            return base64.b64encode(image.read()).decode('utf-8')

    def __image_pil_to_base64(self, image: Image) -> str:
        """
        Converts a PIL Image to base64
        """
        buffer = io.BytesIO()
        image.save(buffer, format = image.format)
        buffer = buffer.getvalue()
        return base64.b64encode(buffer).decode('utf-8')


    def get_json_response(self, image_path: str) -> str:
        """
        Returns JSON message with encoded image
        """
        encoded = self.__image_path_to_base64(image_path)

        message = {
            "returned_image": encoded
        }
        return json.dumps(message)

    def get_json_response_pil(self, image: Image) -> str:
        """
        Returns JSON message with base64 encoded image from PIL Image
        """
        encoded = self.__image_pil_to_base64(image)

        message = {
            "returned_image": encoded
        }
        return json.dumps(message)

#ser = ServerUtil()
#print(ser.get_json_response("data/image.jpg"))
