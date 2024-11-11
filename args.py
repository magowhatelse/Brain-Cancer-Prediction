import argparse

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--backbone",
                        type=str,
                        choices=["resnet18", "resnet34", "resnet50"],
                        default="resnet34")
    # define output directory (model results, weights)
    parser.add_argument("--out_dir", "--out_dir", type=str, default="session") 
    # CSV 
    parser.add_argument("--csv_dir", "--csv_dir", default=r"data/CSV")

    parser.add_argument("--batch_size", "--batch_size", type=int, default=16,
                        choices=[16, 32, 64])
    
    parser.add_argument("--learning_rate", "--learning_rate", type=int, default=0.001,
                        choices=[0.0001, 0.00001, 0.005])
    
    parser.add_argument("--epochs", "--epochs", type=int, default=100,
                        choices=[100, 500, 1000])


    args = parser.parse_args()

    return args
