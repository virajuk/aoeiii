import pandas as pd
import glob
import os

df = pd.read_csv("/home/viraj-uk/Pictures/th_3_british/th_3_british.csv")

images = glob.glob("/home/viraj-uk/Pictures/th_3_british/train/*.jpeg")
df2 = df.copy()

for image in images:

    # print(image)
    if os.path.basename(image) != "1.jpeg":
        df2['filename'] = os.path.basename(image)
        df = df.append(df2, ignore_index = True)
        # print(image)

df.to_csv("/home/viraj-uk/Pictures/th_3_british/train/th_3_british_train.csv", index=False)