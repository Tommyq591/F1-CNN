import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, model, config, class_names, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.config = config
        self.class_names = list(class_names)
        # One mutually exclusive team label per image: feed raw logits to cross entropy.
        self.loss_fn = nn.CrossEntropyLoss()
        # L2 weight decay penalises large weights; Adam adapts each parameter's step size.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
        self.history = []
        self.best_epoch = None

    def _epoch(self, loader, training):
        self.model.train(training)
        total_loss, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(training):
            for images, targets in loader:
                images, targets = images.to(self.device), targets.to(self.device)
                if training:
                    # Gradients accumulate by default, so clear them for each mini-batch.
                    self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)
                if training:
                    # Backpropagate the loss, then apply the optimiser's parameter update.
                    loss.backward()
                    self.optimizer.step()
                # Weight batch means by batch size, including the smaller last batch.
                total_loss += loss.item() * targets.size(0)
                correct += (logits.argmax(1) == targets).sum().item()
                count += targets.size(0)
        if not count:
            raise ValueError('Cannot train or evaluate an empty loader.')
        return total_loss / count, correct / count

    def fit(self, train_loader, validation_loader, output_dir):
        if self.config.epochs < 1:
            raise ValueError('epochs must be at least one.')
        output = Path(output_dir)
        # Separate experiment directories prevent accidental replacement of results.
        output.mkdir(parents=True, exist_ok=False)
        best_loss, best_state = float('inf'), None
        stale_epochs = 0
        for epoch in range(1, self.config.epochs + 1):
            started = time.perf_counter()
            train_loss, train_acc = self._epoch(train_loader, True)
            val_loss, val_acc = self._epoch(validation_loader, False)
            row = dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
                       train_accuracy=train_acc, validation_accuracy=val_acc,
                       seconds=time.perf_counter() - started)
            self.history.append(row)
            print(row, flush=True)
            if val_loss < best_loss:
                stale_epochs = 0
                self.best_epoch = epoch
                best_loss = val_loss
                # Copy tensors so later optimiser steps cannot mutate the saved best weights.
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save({'state_dict': best_state, 'class_names': self.class_names,
                            'config': self.config.to_dict(), 'epoch': epoch}, output / 'best.pt')
            else:
                stale_epochs += 1
            (output / 'history.json').write_text(json.dumps(self.history, indent=2))
            if stale_epochs >= self.config.patience:
                print(f'Early stopping; restoring epoch {self.best_epoch}.', flush=True)
                break
        if best_state is None:
            raise RuntimeError('No finite validation loss. Check images and learning rate.')
        self.model.load_state_dict(best_state)
        return self.history

    def evaluate(self, loader):
        self.model.eval()
        truth, predicted = [], []
        with torch.no_grad():
            for images, targets in loader:
                predicted.extend(self.model(images.to(self.device)).argmax(1).cpu().tolist())
                truth.extend(targets.tolist())
        if not truth:
            raise ValueError('Cannot evaluate an empty loader.')
        labels = list(range(len(self.class_names)))
        return {'accuracy': accuracy_score(truth, predicted),
                'balanced_accuracy': balanced_accuracy_score(truth, predicted),
                'macro_f1': f1_score(truth, predicted, labels=labels, average='macro', zero_division=0),
                'truth': truth, 'predicted': predicted,
                'classification_report': classification_report(truth, predicted, labels=labels,
                    target_names=self.class_names, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(truth, predicted, labels=labels).tolist()}
