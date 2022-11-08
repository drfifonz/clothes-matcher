import pandas as pd


class LandmarkUtils:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_file_list(self, running_mode: str) -> pd.DataFrame:  # TODO consider output as a list

        data = self.__load_evaluation_list(self.path)

        match running_mode:
            case "train":
                data = data[data["evaluation_status"] == running_mode]

                return data["image_name"]

            case "test":
                data = data[data["evaluation_status"] == running_mode]

                return data["image_name"]
            case "val":
                data = data[data["evaluation_status"] == running_mode]

                return data["image_name"]
            case other:
                print("No accetable runnig mode for selecting dataset.")

    def __load_evaluation_list(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path, sep=" ", skiprows=[0, 1], names=["image_name", "evaluation_status"])

        return data
