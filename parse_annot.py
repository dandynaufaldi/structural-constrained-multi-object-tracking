import os
import time

import cv2
import pandas as pd

dataset_dir = os.path.join("dataset", "2DMOT2015", "train")
scenes = os.listdir(dataset_dir)
# <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
columns = ["frame", "id", "left", "top", "width", "height", "conf", "x", "y", "z"]
for scene in scenes:
    scene_path = os.path.join(dataset_dir, scene)
    for kind in ["gt", "det"]:
        gt_path = os.path.join(scene_path, kind)
        gt_label_path = os.path.join(gt_path, "{}.txt".format(kind))
        gt_df = pd.read_csv(gt_label_path, header=None, names=columns)
        parsed_gt_path = os.path.join(gt_path, "{}.csv".format(kind))
        gt_df.to_csv(parsed_gt_path, index=False)
    