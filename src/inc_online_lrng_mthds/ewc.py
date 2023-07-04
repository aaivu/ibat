import torch
import torch.nn as nn
import torch.optim as optim

class EWC():
    def __init__(self, model, dataloader, base_loss, fisher_multiplier):
        self.model = model
        self.dataloader = dataloader
        self.base_loss = base_loss
        self.fisher_multiplier = fisher_multiplier
        self.precision_matrices = {}

        self._compute_fisher()

    def _compute_fisher(self):
        self.model.eval()
        for inputs, labels in self.dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.base_loss(outputs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    fisher = param.grad.data ** 2
                    if name not in self.precision_matrices:
                        self.precision_matrices[name] = fisher
                    else:
                        self.precision_matrices[name] += fisher

    def update_loss(self, criterion):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                precision = self.precision_matrices[name]
                param.grad.data += self.fisher_multiplier * precision * (param - param.data)

    def train(self, criterion, optimizer, num_epochs):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in self.dataloader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                self.update_loss(criterion)  
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss / len(self.dataloader)}")
