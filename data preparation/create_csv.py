import pandas as pd
import os
import seaborn as sns 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

main_dir = r"C:\Brain Cancer Prediction\data\Images"
meta_data = pd.DataFrame(columns=[
    "Name", "Path", "Cancer"
])


for target in os.listdir(main_dir):
    folder_path = os.path.join(main_dir, f"{target}")

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path,image_name)

        meta_data= meta_data._append({
            "Name": image_name,
            "Path": image_path,
            "Cancer": target
        }, ignore_index=True)

meta_data.to_csv(r"C:\Brain Cancer Prediction\data\CSV/data.csv", index=False)

# lengths = [len(meta_data[meta_data["Cancer"] == value]) for value in meta_data["Cancer"].unique()]
# grades = [i for i in range(5)]
# plot = sns.barplot(x=grades,y=lengths).set_title("Distribution of Cancer kinds")
# plt.xlabel("Cancer kind")
# plt.ylabel("Count")
# plt.show()

# #-------------- Train Test split ---------------#
# X = meta_data.drop(columns=["Cancer"], inplace=False)
# y = meta_data["Cancer"]
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# #-------------- Train Val split ---------------#
# X_train, X_valdiation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2) 

# # plots of the distribution
# y_train.value_counts().sort_index().plot(kind="bar")
# plt.savefig("train_distribution")
# y_test.value_counts().sort_index().plot(kind="bar")
# plt.savefig("test_distribution")
# y_validation.value_counts().sort_index().plot(kind="bar")
# plt.savefig("validation_distribution")
