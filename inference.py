# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import time
import random
import os
import math
from scipy.stats import pearsonr


def sensitivity_pruning(model, threshold=1e-3):
    for name, param in model.named_parameters():
        if 'dropout' not in name:

            sensitivity = torch.norm(param.grad)

            if sensitivity < threshold:
                param.data = torch.zeros_like(param.data)


class EarlyStopping:
    def __init__(self, patience=5, delta=0):

        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None or score > self.best_score:
            self.counter = 0
            self.best_score = score


        elif score < self.best_score + self.delta:
            self.counter += 1
            print(self.counter)

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.counter = 0
            self.best_score = score


class ATU(nn.Module):
    def __init__(self, input_dim, alpha_initializer='ones', beta_initializer='zeros',
                 alpha_regularizer=None, beta_regularizer=None,
                 alpha_constraint=None, beta_constraint=None):
        super(ATU, self).__init__()
        self.alpha_initializer = alpha_initializer
        self.beta_initializer = beta_initializer
        self.alpha_regularizer = alpha_regularizer
        self.beta_regularizer = beta_regularizer
        self.alpha_constraint = alpha_constraint
        self.beta_constraint = beta_constraint

        self.alpha = nn.Parameter(self.get_initializer(alpha_initializer)(torch.Tensor(input_dim)))
        self.beta = nn.Parameter(self.get_initializer(beta_initializer)(torch.Tensor(input_dim)))

    def get_initializer(self, initializer):
        if initializer == 'ones':
            return nn.init.ones_
        elif initializer == 'zeros':
            return nn.init.zeros_
        else:
            raise NotImplementedError(f" {initializer} ")

    def forward(self, x):
        outputs = self.alpha * x + self.beta
        return outputs


class TAAF(nn.Module):
    def __init__(self, input_dim, activation):
        super(TAAF, self).__init__()
        self.bottom = ATU(input_dim)
        self.activation = nn.Tanh()
        self.top = ATU(input_dim)

    def forward(self, x):
        x = self.bottom(x)
        x = self.activation(x)
        x = self.top(x)
        return x


class TAAFDense(nn.Module):
    def __init__(self, in_size, n_hidden, activation):
        super(TAAFDense, self).__init__()
        self.dense = nn.Linear(in_size, n_hidden, bias=False)
        self.adaptive_activation = TAAF(n_hidden, activation)

    def forward(self, x):
        x = self.dense(x)
        x = self.adaptive_activation(x)
        return x


class DropoutMLP(nn.Module):
    DROPOUT_RATE = 0.5
    OUTPUT_INIT = 1e-4

    def __init__(self, in_size, n_hidden, out_size):
        super(DropoutMLP, self).__init__()

        self.hidden1 = TAAFDense(in_size, n_hidden, 'Tanh')
        self.batch_norm1 = nn.BatchNorm1d(n_hidden)
        self.dropout1 = nn.Dropout2d(self.DROPOUT_RATE)

        self.hidden2 = TAAFDense(n_hidden + in_size, n_hidden, 'Tanh')
        self.batch_norm2 = nn.BatchNorm1d(n_hidden)

        self.dropout2 = nn.Dropout2d(self.DROPOUT_RATE)

        self.hidden3 = TAAFDense(2 * n_hidden + in_size, n_hidden, 'Tanh')
        self.batch_norm3 = nn.BatchNorm1d(n_hidden)

        self.dropout3 = nn.Dropout2d(self.DROPOUT_RATE)

        self.output = TAAFDense(3 * n_hidden + in_size, out_size, 'Tanh')

    def forward(self, x):
        a = self.hidden1(x)
        a = self.batch_norm1(a)
        a = self.dropout1(a)
        dense_input1 = torch.cat((x, a), dim=1)
        b = self.hidden2(dense_input1)
        b = self.batch_norm2(b)
        b = self.dropout2(b)
        dense_input2 = torch.cat((x, a, b), dim=1)
        c = self.hidden3(dense_input2)
        c = self.batch_norm3(c)
        c = self.dropout3(c)
        dense_input3 = torch.cat((x, a, b, c), dim=1)
        output = self.output(dense_input3)

        return output


def main():
    base_name = sys.argv[1]
    n_epoch = int(sys.argv[2])
    n_hidden = int(sys.argv[3])
    n_MOMENTUM = int(sys.argv[4])
    n_EarlyStopping = int(sys.argv[5])
    n_in_size = int(sys.argv[6])
    n_out_size = int(sys.argv[7])

    in_size = n_in_size
    out_size = n_out_size
    b_size = 200

    MOMENTUM = n_MOMENTUM
    LEARNING_RATE_FACTOR = 1e-2
    # 5E-4
    LEARNING_RATE_START = 5e-4 * LEARNING_RATE_FACTOR
    LEARNING_RATE_MIN = 1e-5 * LEARNING_RATE_FACTOR
    LEARNING_RATE_DECAY = 0.9
    DROPOUT_LEARNING_SCALE = 3

    if not os.path.exists(base_name):
        os.makedirs(base_name)

    print('loading data...')

    X_tr = np.load('/your_X_train_dataset')
    Y_tr = np.load('/your_Y_train_dataset')
    Y_tr_target = np.array(Y_tr)
    X_va = np.load('/your_X_va_dataset')
    Y_va = np.load('/your_Y_va_dataset')
    Y_va_target = np.array(Y_va)
    X_te = np.load('/your_X_te_dataset')
    Y_te = np.load('/your_Y_te_dataset')
    Y_te_target = np.array(Y_te)

    model = DropoutMLP(in_size, n_hidden, out_size).cuda()
    training_loss_func = nn.MSELoss(reduction='sum')
    test_loss_func = nn.L1Loss()
    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if 'dropout1' not in name]},
                           {'params': model.dropout1.parameters(), 'lr': LEARNING_RATE_START * DROPOUT_LEARNING_SCALE}],
                          lr=LEARNING_RATE_START, momentum=MOMENTUM, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LEARNING_RATE_DECAY, patience=0,
                                                           threshold=0, min_lr=LEARNING_RATE_MIN)

    monitor_idx_tr = random.sample(range(88807), 5000)
    X_tr_monitor, Y_tr_monitor_target = X_tr[monitor_idx_tr, :], Y_tr_target[monitor_idx_tr, :]

    train_dataset = TensorDataset(torch.from_numpy(X_tr).float().cuda(), torch.from_numpy(Y_tr).float().cuda())
    train_loader = DataLoader(train_dataset, batch_size=b_size)
    MSE_tr_old = 10.0
    MSE_va_old = 10.0
    MSE_va_best = 10.0
    MSE_te_old = 10.0
    outlog = open('./' + base_name + '/' + base_name + '.log', 'w')
    log_str = '\t'.join(map(str, ['epoch', 'MSE_va', 'MSE_va_change', 'MSE_te', 'MSE_te_change',
                                  'MSE_tr', 'MSE_tr_change', 'Pearson_va', 'Pearson_te',
                                  'learning_rate', 'time(sec)']))
    print(log_str)
    outlog.write(log_str + '\n')
    early_stopping = EarlyStopping(patience=n_EarlyStopping)
    for epoch in range(n_epoch):
        t_old = time.time()
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = training_loss_func(outputs, targets)
            loss.backward()
            sensitivity_pruning(model)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            Y_va_hat = model(torch.from_numpy(X_va).float().cuda()).cpu().numpy()
            Y_te_hat = model(torch.from_numpy(X_te).float().cuda()).cpu().numpy()
            Y_tr_hat_monitor = model(torch.from_numpy(X_tr_monitor).float().cuda()).cpu().numpy()
            pearson_corr_va = pearsonr(Y_va_hat.flatten(), Y_va_target.flatten())[0]
            pearson_corr_te = pearsonr(Y_te_hat.flatten(), Y_te_target.flatten())[0]

        MSE_va = np.square(Y_va_target - Y_va_hat).mean()
        MSE_te = np.square(Y_te_target - Y_te_hat).mean()
        MSE_tr = np.square(Y_tr_monitor_target - Y_tr_hat_monitor).mean()

        MSE_va_change = (MSE_va - MSE_va_old) / MSE_va_old
        MSE_te_change = (MSE_te - MSE_te_old) / MSE_te_old
        MSE_tr_change = (MSE_tr - MSE_tr_old) / MSE_tr_old

        MSE_va_old = MSE_va
        MSE_te_old = MSE_te
        MSE_tr_old = MSE_tr

        t_new = time.time()
        l_rate = optimizer.param_groups[0]['lr']
        log_str = '\t'.join(map(str, [epoch + 1, '%.6f' % MSE_va, '%.6f' % MSE_va_change, '%.6f' % MSE_te,
                                      '%.6f' % MSE_te_change, '%.6f' % MSE_tr, '%.6f' % MSE_tr_change,
                                      '%.6f' % pearson_corr_va, '%.6f' % pearson_corr_te,
                                      '%.8f' % l_rate, int(t_new - t_old)]))

        print(log_str)
        outlog.write(log_str + '\n')
        outlog.flush()

        scheduler.step(MSE_tr)
        optimizer.param_groups[0]['lr'] = l_rate
        early_stopping(-MSE_va, model)

        if early_stopping.early_stop:
            print("EARLY STOP!")
            break

        if MSE_va < MSE_va_best:
            MSE_va_best = MSE_va

            torch.save(model.state_dict(), './' + base_name + '/' + base_name + '_bestva_model.pth')

    print('MSE_va_best: %.6f' % (MSE_va_best))

    outlog.write('MSE_va_best : %.6f' % (MSE_va_best) + '\n')
    outlog.close()


if __name__ == '__main__':
    main()
