import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MRI_dataset
from model import MyModel
from args import get_args
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay, precision_recall_curve, auc
import matplotlib.pyplot as plt
import torch.optim as optim

def helper(model, test_loader, device):
    model.eval()

    y_pred = []
    y_true = []
    y_pred_soft = []

    test_running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["img"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)
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

    # fpr, tpr, _ = precision_recall_curve(y_true, y_pred_soft[:, 1])
    # pr_auc = auc(fpr, tpr)

    return accuracy, balanced_accuracy, roc_auc, precision, y_pred, y_true, y_pred_soft

def evaluate_model():
    args = get_args()
    model_name = args.backbone

    test_set = pd.read_csv(r"C:\Brain Cancer Prediction\data\CSV\test_data.csv")
    test_dataset = MRI_dataset(dataset=test_set)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    folds_y_pred = []
    folds_y_true = []
    folds_y_pred_soft = []

    for fold in range(5):
        print('Fold: ', fold)
        path_weights_fold = os.path.join(args.out_dir, model_name + fr'__fold_{fold}.pth')

        model = MyModel().to(device)
        model.load_state_dict(torch.load(path_weights_fold), strict=False)

        accuracy, balanced_accuracy, roc_auc, precision, y_pred, y_true, y_pred_soft = helper(model, test_loader, device)

        print(f'Accuracy of the network: {accuracy} %')
        print(f"Balanced Accuracy: {balanced_accuracy}, ROC-AUC-Score: {roc_auc},  Precision: {precision}")

        folds_y_pred.extend(y_pred)
        folds_y_true.extend(y_true)
        folds_y_pred_soft.extend(y_pred_soft)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()

        out_dir = "C:\Brain Cancer Prediction\data exploration\plots"
        plot_filename = os.path.join(out_dir, f"{model_name}_confusion_matrix_{fold}.png")
        plt.savefig(plot_filename)

        break

  
  

    # # Plot the ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % overall_roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # plt.savefig(os.path.join(out_dir, "roc_curve.png"))

    # Plot the Precision-Recall curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(tpr, [x[1] for x in folds_y_pred_soft], color='darkorange', lw=2, label='PR curve (area = %0.2f)' % overall_pr_auc)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc="lower left")
    # plt.savefig(os.path.join(out_dir, "pr_curve.png"))

if __name__ == '__main__':
    evaluate_model()