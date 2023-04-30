import os
import json
from itertools import combinations
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with open("NAB/labels/combined_labels.json") as label_file:
    labels = json.load(label_file)


dir = "NAB/data"
for folder in os.listdir(dir):
    if "." not in folder:
        subdir = dir + "/" + folder
        c = 0
        for file, file_labels in labels.items():
            c += 1
            if file.startswith(folder):
                full_path = dir + "/" + file
                df = pd.read_csv(full_path)
                print("df load success: ", full_path)
                df["label"] = np.zeros(
                    len(df)
                )
                for label in file_labels:
                    idx = df[df["timestamp"] == label].index
                    df.loc[
                        idx, "label"
                    ] = 1

                df.to_csv(dir + "/" + file + "_labeled.csv", index=False)

