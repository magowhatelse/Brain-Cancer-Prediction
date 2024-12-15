import os
import torch 
import torch.nn as nn
from torch import optim
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay
import numpy as np
from torcheval.metrics import MulticlassAUROC, MulticlassAUPRC, MulticlassConfusionMatrix, MulticlassAccuracy
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from args import get_args

class Trainer:
    def __init__(self, model, train_loader, val_loader, args, fold):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.fold = fold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        # Initialize metric trackers
        self.best_balanced_acc = 0.0
        self.best_model_path = ''
        
    def train_epoch(self):
        """method to train one epoch

        Returns:
            metrics: loss, acc, balanced acc, roc auc,average precison 
        """
        self.model.train()
        running_loss = 0.0
        y_pred, y_true = [], []
        all_roc_scores, all_prc_scores = [], []
        
        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            inputs = batch["img"].to(self.device)
            targets = batch["target"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs_softmax = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs_softmax, 1)
            
            # Calculate loss and backpropagate
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            # Calculate batch metrics
            train_roc = MulticlassAUROC(num_classes=3, average=None)
            train_roc.update(outputs_softmax, targets)
            all_roc_scores.append(train_roc.compute().mean().item())
            
            train_prc = MulticlassAUPRC(num_classes=3, average="macro")
            train_prc.update(outputs_softmax, targets)
            all_prc_scores.append(train_prc.compute().item())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ROC': f"{np.mean(all_roc_scores):.4f}"
            })
        
        # Calculate epoch metrics
        epoch_metrics = {
            'loss': running_loss / len(self.train_loader),
            'balanced_acc': balanced_accuracy_score(y_true, y_pred),
            'roc_auc': np.mean(all_roc_scores),
            'avg_precision': np.mean(all_prc_scores)
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self):
        """ method to validate 

        Returns:
            validation metrics: loss, acc, balanced acc, roc auc,average precison
        """
        self.model.eval()
        running_loss = 0.0
        y_pred, y_true, y_pred_soft = [], [], []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            inputs = batch["img"].to(self.device)
            targets = batch["target"].to(self.device)
            
            outputs = self.model(inputs)
            outputs_softmax = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs_softmax, 1)
            
            loss = self.criterion(outputs, targets)
            running_loss += loss.item()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_soft.extend(outputs_softmax.cpu().numpy())
        
        # Calculate validation metrics
        val_metrics = {
            'loss': running_loss / len(self.val_loader),
            'balanced_acc': balanced_accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_soft, multi_class="ovo", average="macro"),
            'avg_precision': average_precision_score(y_true, y_pred_soft, average="macro")
        }
        
        return val_metrics
    
    def train(self, fold):
        """ main train loop

        Args:
            fold : 1-5

        Returns:
            dataframe: train and validation metrics
        """
        model_name = "VGG16"
        metrics_history = {
            'train_loss': [], 'train_balanced_acc': [],      
            'train_roc_auc': [], 'train_avg_precision': [],
            'val_loss': [], 'val_balanced_acc': [], 
            'val_roc_auc': [], 'val_avg_precision': []
        }
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['balanced_acc'])
            
            # Save best model
            args = get_args()
            if val_metrics['balanced_acc'] > self.best_balanced_acc:
                self.best_balanced_acc = val_metrics['balanced_acc']
                self.best_model_path = f'{self.args.out_dir}/{model_name}__fold_{fold}.pth'
                torch.save(self.model.state_dict(), self.best_model_path)
            
            # Update metrics history
            for key in train_metrics:
                metrics_history[f'train_{key}'].append(train_metrics[key])
                metrics_history[f'val_{key}'].append(val_metrics[key])
            
            # Print epoch summary
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train Acc: {train_metrics['balanced_acc']:.4f} | Val Acc: {val_metrics['balanced_acc']:.4f}")
            print(f"Train ROC: {train_metrics['roc_auc']:.4f} | Val ROC: {val_metrics['roc_auc']:.4f}")
        
        # Save final metrics
        metrics_df = pd.DataFrame(metrics_history)
        
        return metrics_df

    
   