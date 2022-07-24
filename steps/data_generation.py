import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from villard import V
from villard.io import BaseDataReader


def generate_single_image(
    image_key_name: str, image_size: int, object_size: int, n_objects: int
) -> Image:
    obj_img = V.read_data(image_key_name)
    # create empty image with size of image_size
    imarray = np.random.rand(image_size, image_size, 3) * (255 * 0.5)
    img_empty = Image.fromarray(imarray.astype("uint8")).convert("RGBA")
    for i in range(n_objects):
        obj_img = obj_img.resize((object_size, object_size))
        obj_img = obj_img.rotate(np.random.randint(0, 360))
        # resize randomly
        obj_img = obj_img.resize(
            (
                np.random.randint(object_size // 2, object_size),
                np.random.randint(object_size // 2, object_size),
            )
        )
        # paste obj at random position in img_empty
        x = np.random.randint(0, image_size - object_size)
        y = np.random.randint(0, image_size - object_size)
        img_empty.paste(obj_img, (x, y), obj_img)
    # convert to RGB
    img_empty = img_empty.convert("RGB")
    return img_empty


def generate_n_images(
    out_dir: str,
    n_classes: int,
    n_per_class: int,
    n_views: int,
    image_size: int,
    object_size: int,
) -> None:
    obj_names = os.listdir("data/01_raw/generator")
    obj_names = [o.split(".")[0] for o in obj_names]

    # the `class_` is an integer indicating the count of obj in
    # the image.
    n = 0
    dataset_meta = []
    for count in range(1, n_classes + 1):
        for obj_name in obj_names:
            for i in range(n_per_class):
                n += 1
                filenames = []
                for view in range(n_views):
                    img = generate_single_image(
                        obj_name, image_size, object_size, count
                    )
                    filename = os.path.join(out_dir, f"img_{n}_view_{view + 1}.png")
                    filenames.append(filename)
                    img.save(filename)

                row = [obj_name, count] + filenames
                dataset_meta.append(row)
    df_dataset_meta = pd.DataFrame(dataset_meta)
    df_dataset_meta.columns = ["obj_name", "count"] + [
        f"view_{i + 1}" for i in range(n_views)
    ]
    df_dataset_meta.to_csv(
        os.path.join(out_dir, "generated_dataset_metadata.csv"), index=False
    )


@V.node("generate_train_data")
def generate_train_data(
    n_classes: int, n_per_class: int, n_views: int, image_size: int, object_size: int
) -> None:
    generate_n_images(
        "data/02_intermediate/generated_train/",
        n_classes,
        n_per_class,
        n_views,
        image_size,
        object_size,
    )


@V.node("generate_test_data")
def generate_test_data(
    n_classes: int, n_per_class: int, n_views: int, image_size: int, object_size: int
) -> None:
    generate_n_images(
        "data/02_intermediate/generated_test/",
        n_classes,
        n_per_class,
        n_views,
        image_size,
        object_size,
    )
