import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np

main_dir = r"C:\Brain Cancer Prediction\data\Images"
out_dir = r"C:\Brain Cancer Prediction\data\CSV/data.csv"
output_dir = r"C:\Brain Cancer Prediction\data\CSV"

def create_csv_and_folds():
    meta_data = pd.DataFrame(columns=[
        "Name", "Path", "Cancer"
    ])

    for target in range(3):
        folder_path = os.path.join(main_dir, f"{target}")
        print(f"Files in {folder_path}: {len(os.listdir(folder_path))}")

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path,image_name)

            meta_data= meta_data._append({
                "Name": image_name,
                "Path": image_path,
                "Cancer": target
            }, ignore_index=True)

    meta_data.to_csv(out_dir, index=False)


    #-------------- Train Test split ---------------#
    X = meta_data.drop(columns=["Cancer"], inplace=False)
    y = meta_data["Cancer"]
    y = pd.Series(y, dtype=np.int64)

    assert len(X) == len(y), "X and y lengths do not match!"

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    # save test set to dir
    test_data = pd.concat([X_test, y_test], axis=1)
    path_train = os.path.join(output_dir, "test_data.csv")
    test_data.to_csv(path_train ,index=False)

    # save train set
    train_data = pd.concat([X_train, y_train], axis=1)
    path_train = os.path.join(output_dir, "train_data.csv")
    train_data.to_csv(path_train ,index=False)

    # reset index
    X = X_train.reset_index(drop=True)
    y = y_train.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # iterate over the stratified splits
    fold = 0
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        # create train data fold
        train_data = pd.concat([X_train, y_train], axis=1)
        path_train = os.path.join(output_dir, f"fold_{fold}_train.csv")
        train_data.to_csv(path_train ,index=False)

        # create val data fold
        val_data = pd.concat([X_test, y_test], axis=1)
        path_val = os.path.join(output_dir,f"fold_{fold}_val.csv")
        val_data.to_csv(path_val, index=False)

        fold += 1

if __name__ == '__main__':
    create_csv_and_folds()