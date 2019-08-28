import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import sys

_class = ['spikes_gnm2', 'spikes_nm2', 'spikes_nnm2', 'spontaneous1']

# ------------ Dataset -------------------
class Monkey_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, num=30, iso=True, split=0.6, train=True):
        super().__init__()
        mode = 'iso' if iso else 'pca'
#         self.datas = np.load(root + f'{mode}inputs_all.npy')
#         self.labels = np.load(root + f'{mode}labels_all_index.npy')
        datas = np.load(root + f'{mode}_random.npy')
        split = int(0.6 * datas.shape[0])
        self.datas = datas[:split] if train else datas[split:]

    def __len__(self):
        """
            return: the number of sample(int type)
        """
        return self.datas.shape[0]

    def __getitem__(self, index):
        """
        return: tensor [1, embedding] and LongTensor [1]
        """
        _data = self.datas[index]
        data, label = _data[:30], _data[30]
        # crop
        data = data[:num]
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.LongTensor([int(label)])
        return data, label
    
# ------------ Network -------------------
class Network(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_feature, 128, bias=True)
        # self.bn1 = torch.nn.BatchNorm1d(256)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(128, 256, bias=True)
        # self.bn2 = torch.nn.BatchNorm1d(128)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.linear3 = torch.nn.Linear(256, out_feature, bias=True)

    def forward(self, x):
        # linear64 -> relu -> dropout0.2 -> linear256 -> relu -> dropout0.2 -> linear4
        x = self.linear1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return x
    
# ------------ Lr Scheduler -------------------
def adjust_lr(epoch, epochs, lr_init, optimizer):
    if epoch < (epochs / 4 * 1):
        _lr = lr_init
    elif epoch < (epochs / 4 * 2):
        _lr = lr_init * 0.1
    elif epoch < (epochs / 4 * 3):
        _lr = lr_init * 0.01
    else:
        _lr = lr_init * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr
    return _lr
#     print('lr is changed to ', _lr)


#
class Average:
    def __init__(self):
        self.sum = 0.0
        self.idx = 0.0
    @property
    def score(self):
        return self.sum / self.idx
    
# setup
num = 10
class_num = 4
cuda_index = 2
epochs = 100
lr_init = 0.01
# initilize dataset class
monkey_dataset_train = Monkey_Dataset('', num=num, iso=False, train=True)
monkey_dataset_val = Monkey_Dataset('', num=num, iso=False, train=False)
# initilize dataloader class
monkey_train = torch.utils.data.DataLoader(monkey_dataset_train, batch_size=256, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
monkey_val = torch.utils.data.DataLoader(monkey_dataset_val, batch_size=128, shuffle=False, num_workers=10, pin_memory=True, drop_last=False)
# initilize network class
network = Network(num, class_num)
network.cuda(cuda_index)
# initilize criterion class
criterion = torch.nn.CrossEntropyLoss()
criterion.cuda(cuda_index)
# initilize optimizer class
optimizer = torch.optim.Adam(network.parameters(), lr=lr_init)
# optimizer = torch.optim.SGD(network.parameters(), lr=lr_init, momentum=0.9, weight_decay=5e-4)

# main procedure
sys.stdout.flush()
for epoch in range(epochs):
    # train
    network.train()
    _loss = Average()
    _accuracy = Average()
    pbar = tqdm(monkey_train)
    lr = lr_init
    # lr = adjust_lr(epoch, epochs, lr_init, optimizer)
    for i, (data, label) in enumerate(pbar):
        data, label  = data.cuda(cuda_index), label.cuda(cuda_index)
        label = label.squeeze()
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, label)

        _loss.sum += loss.item()
        _loss.idx += 1.0
        _accuracy.sum += len(torch.nonzero((torch.argmax(output, dim=1, keepdim=False) & label))) / len(label)
        _accuracy.idx += 1.0

        pbar.set_description(f'Training: Epoch: {epoch}, LR: {lr}, loss: {_loss.score:.5f}, accuracy: {(_accuracy.score * 100):.5f}', )
        loss.backward()
        optimizer.step()
#         f.write(f'Training: Epoch: {epoch}, LR: {lr}, loss: {_loss.score:.5f}, accuracy: {(_accuracy.score * 100):.5f}\n')
    pbar.close()

    # val
    with torch.no_grad():
        network.eval()
        _loss = Average()
        _accuracy = Average()
        pbar = tqdm(monkey_val)
#         lr = adjust_lr(epoch, epochs, lr_init, optimizer)
        for i, (data, label) in enumerate(pbar):
            data, label  = data.cuda(cuda_index), label.cuda(cuda_index)
            label = label.squeeze(dim=1)
            output = network(data)
            loss = criterion(output, label)

            _loss.sum += loss.item()
            _loss.idx += 1.0
            _accuracy.sum += len(torch.nonzero((torch.argmax(output, dim=1, keepdim=False) & label))) / len(label)
            _accuracy.idx += 1.0
            pbar.set_description(f'Val: Epoch: {epoch}, LR: {lr}, loss: {_loss.score:.5f}, accuracy: {(_accuracy.score * 100):.5f}', )
        # f.write(f'Val: Epoch: {epoch}, LR: {lr}, loss: {_loss.score:.5f}, accuracy: {(_accuracy.score * 100):.5f}')
        pbar.close()


