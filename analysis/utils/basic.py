import os
from typing import *

__all__ = ["CorrInfo", "Path", "get_file_number"]


def get_file_number(path: str) -> Optional[int]:
    """Get the file number.

    Args:
        path (str): The path of the file.

    Returns:
        int: The file number.
    """
    filename = os.path.basename(path)
    no_ext = os.path.splitext(filename)[0]
    info = no_ext.split("_")
    file_number = int(info.pop(-1)) if info[-1].isdigit() else None
    return file_number


# class CorrInfo:
#     path: str
#     dirname: str
#     file_number: int

#     def __init__(self, path: str) -> "CorrInfo":
#         """Get the corr info from the path of the corrected frames file.

#         Args:
#             path (str): The path of the corrected frames file.

#         Returns:
#             CorrInfo: The corrected frames info.
#         """
#         dirname = os.path.dirname
#         file_number_dirname = os.path.join(dirname, "..")
#         self.path = path
#         self.file_number = int(os.path.basename(file_number_dirname))
#         self.dirname = os.path.join(file_number_dirname, "..", "..")


class Path:
    """The class for the path and directory of the files and folders in the analysis.

    Attributes:
        path: The path of the data file.
    """

    FRAMETIMES_BASE_FILENAME = "frameTimes.mat"
    GENERATE_FOLDER_NAME = "generate"
    NAN_FOLDER_NAME = "NaN"
    LOG_FOLDER_NAME = "log"
    SAMPLING_RATE_FILENAME = "sampling_rate.txt"
    ANCHOR_FILENAME = "anchor.yaml"
    CORRECTED_FRAMES_FILENAME = "corrected_frames.npy"
    RESULT_FOLDER_NAME = "result"
    STRUCTURES_CHANGE_FILENAME = "structures_change.csv"
    IMAGE_FOLDER_NAME = "image"
    IMAGE_FILENAME = "image.png"
    STIMULUS_FILENAME = "stimulus.csv"
    RANGE_FILENAME = "range.txt"
    ROI_FOLDER_NAME = "roi"
    ROI_CHANGE_FILENAME = "roi_change.csv"
    ROI_BOUND_FILENAME = "roi_bound.csv"
    ROI_IMAGE_FILENAME = "roi_image.png"
    ROI_RANGE_FILENAME = "roi_range.txt"
    # TODO: Make a baseline file path for each image and roi. With 2 keys
    # of Mode "Frame Mode"(Frame) and "PreStim Mode"(Second).
    BASELINE_FILENAME = "baseline.txt"

    # def __init__(self, path: str, corr_info: Optional[CorrInfo]) -> None:
    def __init__(self, path: str) -> None:
        """Initialize the path.

        Args:
            path (str): The path of the data file.
        """
        self.raw_path = path

        self.dirname = os.path.dirname(path)
        self.file_number = get_file_number(path)
        # if corr_info is None:
        #     self.dirname = os.path.dirname(path)
        #     self.file_number = get_file_number(path)
        # else:
        #     self.dirname = corr_info.dirname
        #     self.file_number = corr_info.file_number

    @property
    def frametimes_path(self) -> str:
        """The path of the frametimes file."""
        if self.file_number is None:
            return os.path.join(self.dirname, self.FRAMETIMES_BASE_FILENAME)
        else:
            name, ext = os.path.splitext(self.FRAMETIMES_BASE_FILENAME)
            return os.path.join(self.dirname, f"{name}_{self.file_number:04}{ext}")

    @property
    def generate_directory(self) -> str:
        """The directory of the generate folder."""
        directory = os.path.join(self.dirname, self.GENERATE_FOLDER_NAME)
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def file_number_directory(self) -> str:
        """The directory for the generate files from the data file."""
        if self.file_number is None:
            directory = os.path.join(self.generate_directory, self.NAN_FOLDER_NAME)
        else:
            directory = os.path.join(self.generate_directory, f"{self.file_number:04}")
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def log_directory(self) -> str:
        """The directory for the log files."""
        directory = os.path.join(self.file_number_directory, self.LOG_FOLDER_NAME)
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def sampling_rate_path(self) -> str:
        """The path of the sampling rate file."""
        return os.path.join(self.log_directory, self.SAMPLING_RATE_FILENAME)

    @property
    def anchor_path(self) -> str:
        """The path of the file with the anchor for perspective correction."""
        return os.path.join(self.log_directory, self.ANCHOR_FILENAME)

    @property
    def corrected_frames_path(self) -> str:
        """The path of the corrected frames data."""
        return os.path.join(self.log_directory, self.CORRECTED_FRAMES_FILENAME)

    @property
    def result_directory(self) -> str:
        """The directory for the result files."""
        directory = os.path.join(self.file_number_directory, self.RESULT_FOLDER_NAME)
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def structures_change_path(self) -> str:
        """The path of the file with the change of the structures in brain."""
        return os.path.join(self.result_directory, self.STRUCTURES_CHANGE_FILENAME)

    @property
    def image_directory(self) -> str:
        """The directory for the images which need to be saved."""
        directory = os.path.join(self.result_directory, self.IMAGE_FOLDER_NAME)
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def image_number(self) -> int:
        """The image number of the image which needs to be saved."""
        image_number = len(os.listdir(self.image_directory))
        if image_number == 0:
            return image_number
        else:
            image_number -= 1
            image_number_directory = os.path.join(
                self.image_directory, f"{image_number}"
            )
            image_result_path = os.path.join(
                image_number_directory, self.IMAGE_FILENAME
            )
            stim_log_path = os.path.join(image_number_directory, self.STIMULUS_FILENAME)
            range_path = os.path.join(image_number_directory, self.RANGE_FILENAME)
            img_baseline_path = os.path.join(
                image_number_directory, self.BASELINE_FILENAME
            )
            if all(
                [
                    os.path.exists(image_result_path),
                    os.path.exists(stim_log_path),
                    os.path.exists(range_path),
                    os.path.exists(img_baseline_path),
                ]
            ):
                return image_number + 1
            else:
                return image_number

    @property
    def image_number_directory(self) -> str:
        """The directory for the image which needs to be saved."""
        directory = os.path.join(self.image_directory, f"{self.image_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def image_path(self) -> str:
        """The path of the image which needs to be saved."""
        return os.path.join(self.image_number_directory, self.IMAGE_FILENAME)

    @property
    def stimulus_path(self) -> str:
        """The path of the stimulus file attached to the image."""
        return os.path.join(self.image_number_directory, self.STIMULUS_FILENAME)

    @property
    def range_path(self) -> str:
        """The path of the range file attached to the image."""
        return os.path.join(self.image_number_directory, self.RANGE_FILENAME)

    @property
    def img_baseline_path(self) -> str:
        """The path of the baseline file attached to the image."""
        return os.path.join(self.image_number_directory, self.BASELINE_FILENAME)

    @property
    def roi_directory(self) -> str:
        """The directory for the ROI files which need to be saved."""
        directory = os.path.join(self.result_directory, self.ROI_FOLDER_NAME)
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def roi_number(self) -> int:
        """The ROI number of the ROI which needs to be saved."""
        roi_number = len(os.listdir(self.roi_directory))
        if roi_number == 0:
            return roi_number
        else:
            roi_number -= 1
            roi_number_directory = os.path.join(self.roi_directory, f"{roi_number}")
            roi_result_path = os.path.join(
                roi_number_directory, self.ROI_IMAGE_FILENAME
            )
            roi_change_path = os.path.join(
                roi_number_directory, self.ROI_CHANGE_FILENAME
            )
            roi_bound_path = os.path.join(roi_number_directory, self.ROI_BOUND_FILENAME)
            roi_range_path = os.path.join(roi_number_directory, self.ROI_RANGE_FILENAME)
            roi_baseline_path = os.path.join(
                roi_number_directory, self.BASELINE_FILENAME
            )
            roi_stim_path = os.path.join(roi_number_directory, self.STIMULUS_FILENAME)
            if all(
                [
                    os.path.exists(roi_result_path),
                    os.path.exists(roi_change_path),
                    os.path.exists(roi_bound_path),
                    os.path.exists(roi_range_path),
                    os.path.exists(roi_baseline_path),
                    os.path.exists(roi_stim_path),
                ]
            ):
                return roi_number + 1
            else:
                return roi_number

    @property
    def roi_number_directory(self) -> str:
        """The directory for the ROI which needs to be saved."""
        directory = os.path.join(self.roi_directory, f"{self.roi_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def roi_image_path(self) -> str:
        """The path of the ROI image which needs to be saved."""
        return os.path.join(self.roi_number_directory, self.ROI_IMAGE_FILENAME)

    @property
    def roi_change_path(self) -> str:
        """The path of the ROI change file which needs to be saved."""
        return os.path.join(self.roi_number_directory, self.ROI_CHANGE_FILENAME)

    @property
    def roi_bound_path(self) -> str:
        """The path of the ROI bound file which needs to be saved."""
        return os.path.join(self.roi_number_directory, self.ROI_BOUND_FILENAME)

    @property
    def roi_range_path(self) -> str:
        """The path of the ROI range file which needs to be saved."""
        return os.path.join(self.roi_number_directory, self.ROI_RANGE_FILENAME)

    @property
    def roi_baseline_path(self) -> str:
        """The path of the ROI baseline file which needs to be saved."""
        return os.path.join(self.roi_number_directory, self.BASELINE_FILENAME)

    @property
    def roi_stimulus_path(self) -> str:
        """The path of the ROI stimulus file which needs to be saved."""
        return os.path.join(self.roi_number_directory, self.STIMULUS_FILENAME)
