import os


def get_file_number(path: str) -> int:
    filename = os.path.basename(path)
    no_ext = os.path.splitext(filename)[0]
    info = no_ext.split("_")
    file_number = int(info.pop(-1)) if info[-1].isdigit() else None
    return file_number


class Path:
    def __init__(self, path: str) -> None:
        self.path = path

    @property
    def dirname(self) -> str:
        return os.path.dirname(self.path)

    @property
    def generate_directory(self) -> str:
        directory = os.path.join(self.dirname, "generate")
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def file_number(self) -> int:
        return get_file_number(self.path)

    @property
    def file_number_directory(self) -> str:
        if self.file_number is None:
            directory = os.path.join(self.generate_directory, "NaN")
        else:
            directory = os.path.join(self.generate_directory, f"{self.file_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def log_directory(self) -> str:
        directory = os.path.join(self.file_number_directory, "log")
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def result_directory(self) -> str:
        directory = os.path.join(self.file_number_directory, "result")
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def image_directory(self) -> str:
        directory = os.path.join(self.result_directory, "image")
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def sampling_rate_path(self) -> str:
        return os.path.join(self.log_directory, "sampling_rate.txt")

    @property
    def anchor_path(self) -> str:
        return os.path.join(self.log_directory, "anchor.yaml")

    @property
    def corrected_frames_path(self) -> str:
        return os.path.join(self.log_directory, "corrected_frames.npy")

    @property
    def structures_change_path(self) -> str:
        return os.path.join(self.result_directory, "structures_change.npy")

    @property
    def image_number(self) -> int:
        image_number = len(os.listdir(self.image_directory))
        if image_number == 0:
            return image_number
        else:
            image_number -= 1
            image_number_directory = os.path.join(
                self.image_directory, f"{image_number}"
            )
            image_result_path = os.path.join(image_number_directory, "image_result.png")
            stim_log_path = os.path.join(image_number_directory, "stimulus.csv")
            range_path = os.path.join(image_number_directory, "range.txt")
            if all(
                [
                    os.path.exists(image_result_path),
                    os.path.exists(stim_log_path),
                    os.path.exists(range_path),
                ]
            ):
                return image_number + 1
            else:
                return image_number

    @property
    def image_number_directory(self) -> str:
        directory = os.path.join(self.image_directory, f"{self.image_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def image_result_path(self) -> str:
        return os.path.join(self.image_number_directory, "image_result.png")

    @property
    def stimulus_path(self) -> str:
        return os.path.join(self.image_number_directory, "stimulus.csv")

    @property
    def range_path(self) -> str:
        return os.path.join(self.image_number_directory, "range.txt")
