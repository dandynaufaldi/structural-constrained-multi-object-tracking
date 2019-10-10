import os
import time

import cv2
import numpy as np
import pandas as pd

dataset_dir = os.path.join("dataset", "2DMOT2015", "train")
scenes = os.listdir(dataset_dir)
columns = ["frame", "id", "left", "top", "width", "height", "conf", "x", "y", "z"]
bbox_col = ["id", "left", "top", "width", "height"]
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (150, 0, 0),
    (0, 150, 0),
    (0, 0, 150),
    (150, 150, 0),
    (0, 150, 150),
    (150, 0, 150),
    (50, 0, 0),
    (0, 50, 0),
    (0, 0, 50),
    (50, 50, 0),
    (0, 50, 50),
    (50, 0, 50),
]
for scene in scenes:
    scene_path = os.path.join(dataset_dir, scene)
    gt_path = os.path.join(scene_path, "gt", "gt.csv")
    gt_df = pd.read_csv(gt_path)


    images_path = os.path.join(scene_path, "img1")
    image_names = os.listdir(images_path)
    image_names.sort()

    for i, image_name in enumerate(image_names):
        image_path = os.path.join(images_path, image_name)
        img = cv2.imread(image_path)
        frame_id = i + 1
        gt_df_frame = gt_df[gt_df["frame"] == frame_id]
        if len(gt_df_frame) > 0:
            bbox_data = gt_df_frame[bbox_col].values - 1
            bbox_data = bbox_data.astype("int32")
            bbox_data = bbox_data.tolist()
            for (obj_id, left, top, width, height) in bbox_data:
                color_idx = obj_id % len(colors)
                color = colors[color_idx]
                cv2.rectangle(img, (left, top), (left + width, top + height), color, 2)
        cv2.imshow(scene, img)
        key = cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()