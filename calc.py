import os
import time

import cv2
import numpy as np
import pandas as pd

dataset_dir = os.path.join("dataset", "2DMOT2015", "train")
scenes = os.listdir(dataset_dir)
columns = ["frame", "id", "left", "top", "width", "height", "conf", "x", "y", "z"]
bbox_col = ["id", "left", "top", "width", "height"]

for scene in scenes:
    scene_path = os.path.join(dataset_dir, scene)
    gt_path = os.path.join(scene_path, "gt", "gt.csv")
    gt_df = pd.read_csv(gt_path)

    images_path = os.path.join(scene_path, "img1")
    image_names = os.listdir(images_path)

    for i in range(len(image_names)):
        frame_id = i + 1
        gt_df_frame = gt_df[gt_df["frame"] == frame_id]
        if len(gt_df_frame) == 0:
            continue
        bbox_data = gt_df_frame[bbox_col].values - 1
        bbox_data = bbox_data.astype("int32")
        for (_, left, top, width, height) in bbox_data:
            # kalau first frame
            pass
