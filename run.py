import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autoencoder import main

# loop over all yahoo files

if __name__ == "__main__":
    base_dir = "datasets/open-anomaly-detection-benchmark/data/datasets"
    folder = input("Enter the folder of the data (ex. yahoo): ")
    dir = base_dir + "/" + folder

    for file in sorted(os.listdir(dir)):
        print(dir + file)
        main(folder + "/" + file)