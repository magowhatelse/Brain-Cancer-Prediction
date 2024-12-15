import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MRI_dataset
from model import MyModel
from args import get_args
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay, classification_report, auc
import matplotlib.pyplot as plt
import torch.optim as optim



def helper(model, test_loader, device):
    """_summary_

    Args:
        model : trained model
        test_loader
        device: cuda

    Returns:
        results: 
        metrics such as acc, balanced acc and precision,
        roc auc score and the list of the predicted and ground truth
    """
    model.eval()

    y_pred = []
    y_true = []
    y_pred_soft = []

    test_running_loss = 0.0
    correct = 0
    total = 0

    # we dont need the gradients
    with torch.no_grad():

        # iterate over test loader
        for batch in test_loader:
            inputs = batch["img"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)

            # apply softmax function 
            outputs_softmax = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs_softmax, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
            y_pred_soft.extend(outputs_softmax.detach().cpu().numpy())

            loss = nn.CrossEntropyLoss()(outputs, targets)
            test_running_loss += loss.item()

    accuracy = 100 * correct // total
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_soft, multi_class="ovo", average="macro")
    precision = average_precision_score(y_true, y_pred_soft, average="macro")

    return accuracy, balanced_accuracy, roc_auc, precision, y_pred, y_true, y_pred_soft

def evaluate_model():
    """
    function to plot and save the evaluation metric
    """
    args = get_args()
    model_name = "VGG16"

    test_set = pd.read_csv(r"C:\Brain Cancer Prediction\data\CSV\test_data.csv")
    test_dataset = MRI_dataset(dataset=test_set)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    folds_y_pred = []
    folds_y_true = []
    folds_y_pred_soft = []

    # iterate over the 5 folds to get the 5 trained modelss
    for fold in range(5):
        print('Fold: ', fold)

        # get the trained weights of each model for each fold
        path_weights_fold = os.path.join(args.out_dir, model_name + fr'__fold_{fold}.pth')

        # get trained model
        model = MyModel().to(device)
        model.load_state_dict(torch.load(path_weights_fold), strict=False)

        # calc the metrics by calling the helper()
        accuracy, balanced_accuracy, roc_auc, precision, y_pred, y_true, y_pred_soft = helper(model, test_loader, device)

        print(f'Accuracy of the network: {accuracy} %')
        print(f"Balanced Accuracy: {balanced_accuracy}, ROC-AUC-Score: {roc_auc},  Precision: {precision}")

        # add the predicted values and the targets to the list
        folds_y_pred.extend(y_pred)
        folds_y_true.extend(y_true)
        folds_y_pred_soft.extend(y_pred_soft)

        # display the confusion matrix of each model s
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()

        out_dir = r"C:\Brain Cancer Prediction\results\plots"
        plot_filename = os.path.join(out_dir, fr"{model_name}_confusion_matrix_{fold}.png")
        plt.savefig(plot_filename)

        print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    evaluate_model()