import pandas as pd
import os

dataset_list_type = []
dataset_list_img = []

def type_img(name):
    if "notebook" in name:
        return 0
    elif "armario" in name:
        return 2
    elif "cadeira" in name:
        return 3
    return 1

for img in os.listdir("./Dataset"):
    dataset_list_img.append(f"./Dataset/{img}")
    dataset_list_type.append(type_img(img))

df = pd.DataFrame({"Image path": dataset_list_img, "Label": dataset_list_type})
df.to_csv("objects_train.csv", index=False) 