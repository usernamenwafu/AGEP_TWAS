import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from captum.attr import DeepLift
from captum.attr import visualization as viz

from torch.optim import NAdam
import matplotlib.pyplot as plt 
import seaborn as sns


class AutoEncoder(nn.Module):
    def __init__(self, n_inputsize,n_hiddensize):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputsize, n_hiddensize),
            # nn.Dropout(0.1),
            nn.Sigmoid(),
            # nn.BatchNorm1d(6000),
            nn.Linear(n_hiddensize, 100),
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, n_hiddensize),
            nn.Sigmoid(),

            nn.Linear(n_hiddensize, n_inputsize),
        )

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        return x

def main():
    n_epochs = sys.argv[1]
    n_inputsize = sys.argv[2]
    n_hiddensize = sys.argv[3]
    n_top_genes = sys.argv[4]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    X_tr = np.load('/your_X_train_dataset')
    Y_tr = np.load('/your_Y_train_dataset')
    XX_tr = np.concatenate((X_tr, Y_tr), axis=1)
    print("XX_tr:", XX_tr.shape)

    X_va = np.load('/your_X_va_dataset')
    Y_va = np.load('/your_Y_va_dataset')
    XX_va = np.concatenate((X_va, Y_va), axis=1)
    print("XX_va:", XX_va.shape)

    X_te = np.load('/your_X_te_dataset')
    Y_te = np.load('/your_Y_te_dataset')
    XX_te = np.concatenate((X_te, Y_te), axis=1)
    print("XX_te:", XX_te.shape)

    autoencoder = AutoEncoder(n_inputsize=n_inputsize,n_hiddensize=n_hiddensize).to(device)

    criterion = nn.MSELoss()

    batch_size = 64
    train_dataset = TensorDataset(torch.from_numpy(XX_tr).float().cuda(), torch.from_numpy(XX_tr).float().cuda())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(torch.from_numpy(XX_va).float().cuda(), torch.from_numpy(XX_va).float().cuda())
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    optimizer = NAdam(autoencoder.parameters(), lr=0.001,weight_decay=1e-8)

    num_epochs = n_epochs
    for epoch in range(num_epochs):
        autoencoder.train()
        total_train_loss = 0
        for data in train_dataloader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()


        average_train_loss = total_train_loss / len(train_dataloader)


        autoencoder.eval()
        with torch.no_grad():
            total_mse = 0
            predictions = []
            for data in valid_dataloader:
                inputs, _ = data
                outputs = autoencoder(inputs)
                mse = criterion(outputs, inputs)
                total_mse += mse.item()
                predictions.append(outputs.cpu().numpy())

            predictions = np.concatenate(predictions)
            pcc, _ = pearsonr(predictions.flatten(), XX_va.flatten())

            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_train_loss}, Validation MSE: {total_mse / len(valid_dataloader)}, Validation PCC: {pcc}')


    autoencoder.eval()
    with torch.no_grad():
        test_dataset = TensorDataset(torch.from_numpy(XX_te).float().cuda(), torch.from_numpy(XX_te).float().cuda())
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        total_test_mse = 0
        test_predictions = []
        for data in test_dataloader:
            inputs, _ = data
            outputs = autoencoder(inputs)
            mse = criterion(outputs, inputs)
            total_test_mse += mse.item()
            test_predictions.append(outputs.cpu().numpy())

        test_predictions = np.concatenate(test_predictions)
        test_pcc, _ = pearsonr(test_predictions.flatten(), XX_te.flatten())

        print(f'Test MSE: {total_test_mse / len(test_dataloader)}, Test PCC: {test_pcc}')

    autoencoder.eval()

    gene_errors = np.mean((XX_te - test_predictions)**2, axis=1)
    gene_errors_flat = gene_errors.flatten()
    deep_lift = DeepLift(autoencoder)


    test_dataset = TensorDataset(torch.from_numpy(XX_tr).float().cuda())
    test_dataloader = DataLoader(test_dataset, batch_size=1)


    contributions = []
    for data in test_dataloader:
        input_data = data[0]
        contribution = deep_lift.attribute(input_data, target=2)
        contributions.append(contribution.cpu().detach().numpy())

    average_contributions = np.mean(np.array(contributions), axis=0)
    gene_importance = np.argsort(-average_contributions)
    gene_importance = gene_importance.flatten()
    top_genes = gene_importance[:n_top_genes]

    with open('top_genes.txt', 'w') as file:
        file.write("Top genes based on importance:\n")
        for gene_index in top_genes:
            file.write(str(gene_index) + '\n')

    print("Top genes have been saved to 'top_genes.txt' file.")

if __name__ == '__main__':
    main()