import torch

import utils
from LF_1D import LF_Layer
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Geometry_1D import Geometry_1D as G1D
import torch.nn.init as init
SEED = 42


class DiffusionModel(nn.Module):
    def __init__(self,input_dim,geometry_dim,seed=True):
        super(DiffusionModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input = nn.Linear(input_dim,8)
        self.h1 = nn.Linear(8,8)
        self.h2 = nn.Linear(8,8)
        self.h3 = nn.Linear(8, geometry_dim)
        self.out= nn.Hardtanh(min_val=0, max_val=1)
        self._initialize_weights()
        self.loss = nn.MSELoss()
        self.to(self.device)
        if seed:
            torch.manual_seed(SEED)

    def _initialize_weights(self):
        init.kaiming_uniform_(self.input.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.h1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.h2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.h3.weight, nonlinearity='relu')
        init.uniform_(self.input.bias, a=-0.1, b=0.1)
        init.uniform_(self.h1.bias, a=-0.1, b=0.1)
        init.uniform_(self.h2.bias, a=-0.1, b=0.1)
        init.uniform_(self.h3.bias, a=-0.1, b=0.1)
    def forward(self, lengths_list: torch.Tensor, params):
        # w = torch.tensor(params['w'], dtype=torch.float32, requires_grad=True)
        w=params['w']
        if w > 1:
            print('w > 1')
            return
        D_list = []
        for lengths in lengths_list:
            geo = G1D(lengths, params)
            D_numpy = geo.get_D(params['HF_N'])
            D_LF=utils.downsample(D_numpy, params['LF_N'])
            D_tensor = torch.from_numpy(D_LF).float()
            D_list.append(D_tensor)
        D_tensor = torch.stack(D_list).to(self.device)

        x1 = self.input(lengths_list)
        x2 = F.relu(self.h1(x1))
        x3 = F.relu(self.h2(x2))
        fine_D= F.relu(self.h3(x3))
        # fine_D = x
        # if x1.requires_grad:
        #     x1.register_hook(lambda grad: print("fine_D (backward grad):", grad))
        final_D = w * fine_D + (1 - w) * D_tensor
        x = LF_Layer.apply(final_D, params)
        return x

    def load_data(self, lengths_list_train, k_list_train, lengths_list_val, k_list_val, batch_size=32):
        lengths_tensor_train = torch.tensor(lengths_list_train, dtype=torch.float32).to(self.device)
        y_tensor_train = torch.tensor(k_list_train, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(lengths_tensor_train, y_tensor_train)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        lengths_tensor_val = torch.tensor(lengths_list_val, dtype=torch.float32).to(self.device)
        y_tensor_val = torch.tensor(k_list_val, dtype=torch.float32).to(self.device)
        val_dataset = TensorDataset(lengths_tensor_val, y_tensor_val)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        print(f"Training and validation data loaded successfully on {self.device}.")

    def fit(self, params, learning_rate=1,save_path=None):
        epochs=params['epochs']
        train_losses = []
        val_losses = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training Phase
            self.train()
            batch_losses = []
            for x_batch, k_batch in self.train_dataloader:
                pred = self(x_batch,params)
                loss = self.loss(pred, k_batch)
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
                for x_batch, k_batch in self.val_dataloader:
                    pred = self(x_batch,params)
                    loss = self.loss(pred, k_batch)
                    batch_losses.append(loss.item())
            avg_val_loss = np.mean(batch_losses)
            if avg_val_loss>0.1:
                print('Diverge !')
                return None,None,None
            val_losses.append(avg_val_loss)
            # Check for the best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_path is not None:
                    torch.save(self.state_dict(), save_path)
                    print(f"Epoch [{epoch + 1}/{epochs}], New best model saved with Val Loss: {avg_val_loss}")
                else:
                    print(f"Epoch [{epoch + 1}/{epochs}], New best model with Val Loss: {avg_val_loss}")


            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        return train_losses, val_losses,best_val_loss
