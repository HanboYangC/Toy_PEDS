import torch
from LF_1D import LF_Layer
import torch.nn as nn
from torch.utils.data import random_split, TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Geometry_1D import Geometry_1D as G1D
SEED = 42

class DiffusionModel(nn.Module):
    def __init__(self,input_dim,geometry_dim):
        super(DiffusionModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input = nn.Linear(input_dim,64)
        self.h1 = nn.Linear(64,64)
        self.h2 = nn.Linear(64,64)
        self.output = nn.Linear(64,geometry_dim)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self,lengths_list:torch.Tensor,params):
        w=params['w']
        if w>1:
            print('w > 1')
            return
        D_list=[]
        for lengths in lengths_list:
            geo=G1D(lengths,params)
            D_list.append(geo.get_D(params['LF_N']))
        D_tensor = torch.stack(D_list)
        x=self.input(lengths_list)
        x=F.relu(self.h1(x))
        x=F.relu(self.h2(x))
        fine_D=F.hardtanh(self.output(x))
        final_D=w*(fine_D)+(1-w)*D_tensor
        x=LF_Layer.forward(final_D,params)
        return x

    def load_data(self, lengths_list_train, y_list_train, lengths_list_val, y_list_val, lengths_list_test, y_list_test, batch_size=32):
        lengths_tensor_train = torch.tensor(lengths_list_train, dtype=torch.float32).to(self.device)
        y_tensor_train = torch.tensor(y_list_train, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(lengths_tensor_train, y_tensor_train)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        lengths_tensor_val = torch.tensor(lengths_list_val, dtype=torch.float32).to(self.device)
        y_tensor_val = torch.tensor(y_list_val, dtype=torch.float32).to(self.device)
        val_dataset = TensorDataset(lengths_tensor_val, y_tensor_val)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        lengths_tensor_test = torch.tensor(lengths_list_test, dtype=torch.float32).to(self.device)
        y_tensor_test = torch.tensor(y_list_test, dtype=torch.float32).to(self.device)
        test_dataset = TensorDataset(lengths_tensor_test, y_tensor_test)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Training, validation, and test data loaded successfully on {self.device}.")

    def fit(self, params,epochs, learning_rate=0.01):
        train_losses = []
        val_losses = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            # Training Phase
            self.train()
            batch_losses = []
            for x_batch, y_batch in self.train_dataloader:
                pred = self(x_batch,params)
                loss = self.loss(pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            avg_train_loss = np.mean(batch_losses)
            train_losses.append(avg_train_loss)

            # Validation Phase
            self.eval()
            batch_losses = []
            with torch.no_grad():
                for x_batch, y_batch in self.val_dataloader:
                    pred = self(x_batch)
                    loss = self.loss(pred, y_batch)
                    batch_losses.append(loss.item())
            avg_val_loss = np.mean(batch_losses)
            val_losses.append(avg_val_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        return train_losses, val_losses
