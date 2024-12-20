import pandas as pd
import os
from torch.utils.data import DataLoader
import torch 
import time

from args import get_args
from dataset import MRI_dataset
from model import MyModel
from trainer import Trainer
from helper import plot_summary_metrics
from evaluate import evaluate_model

def main():
    """
        main file to run the pipeline
    """

    # 1. step: We need some arguments
    args = get_args()

    # 2. step: iterate over folds
    start_time = time.time()
    for fold in range(5):
        print("Fold: ",fold)

        train_set = pd.read_csv(os.path.join(args.csv_dir ,fr"fold_{fold}_train.csv")) # fold_0_train.csv
        val_set = pd.read_csv(os.path.join(args.csv_dir ,f"fold_{fold}_val.csv"))  

        # 3. step: load dataset
        train_dataset = MRI_dataset(dataset=train_set, is_training=True)
        # train_dataset = MRI_dataset(dataset=train_set, is_training=False)
        val_dataset = MRI_dataset(dataset=val_set)


        # 4. step: create data loaders
        train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,  shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,batch_size=args.batch_size,  shuffle=False)

        # 5. step: init the model
        model = MyModel(backbone=args.backbone)

        # 6. step: train the model 
        t = Trainer(model, train_loader, val_loader,args, fold)
        df= t.train(fold=fold)

        # 7. plot results in each fold
        plot_summary_metrics(df=df, fold=fold, out_dir=r"C:\Brain Cancer Prediction\results\plots", model_name=args.backbone)


    # 7. step: validate the model
    evaluate_model()

    end_time = time.time()
    print("Total time for the 5 folds: ", end_time - start_time)




if "__main__":
    main()