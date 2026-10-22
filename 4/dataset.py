
import kagglehub
import numpy as np
from pathlib import Path


def load_images(file_path):
    with open(file_path, "rb") as f:
        data = np.frombuffer(
            f.read(), dtype=np.uint8, offset=16
        )
    return data.reshape(-1, 28, 28)


def load_labels(file_path):
    with open(file_path, "rb") as f:
        return np.frombuffer(
            f.read(), dtype=np.uint8, offset=8
        )


def load_mnist():
    path = Path(
        kagglehub.dataset_download("hojjatk/mnist-dataset")
    )

    x_train = load_images(path / "train-images.idx3-ubyte")
    y_train = load_labels(path / "train-labels.idx1-ubyte")

    x_test = load_images(path / "t10k-images.idx3-ubyte")
    y_test = load_labels(path / "t10k-labels.idx1-ubyte")

    # Normalizacia pixelov na interval 0 az 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Pridanie obrazoveho kanala
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test
