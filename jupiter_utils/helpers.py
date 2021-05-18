import os
import shutil
from pathlib import Path
import cv2
from matplotlib import pyplot as plt


def get_paths_of_files_by_extension(path, extension='png'):
    return [str(path) for path in list(Path(os.path.abspath(path)).rglob(f"*.{extension}"))]


def print_image(img, title='result'):
    cv2.COLOR_BGRA2RGBA
    copy_img = img.copy()
    copy_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB)
    plt.imshow(copy_img, interpolation='none')
    plt.title(title)
    plt.show()


def create_folder(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
