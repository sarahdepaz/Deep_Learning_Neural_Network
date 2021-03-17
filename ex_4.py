import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as tr, utils
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from matplotlib.legend_handler import HandlerLine2D
from torch.utils import data


BATCH_SIZE = 64
IMAGE_SIZE = 28 * 28
LEARN_RATE = 0.001
EPOCHS = 10
FIRST_HIDDEN_LAYER_SIZE = 100
SECOND_HIDDEN_LAYER_SIZE = 50
OUTPUT_LAYER_SIZE = 10


class ModelA(nn.Module):
    def __init__(self, image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER_SIZE)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(SECOND_HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelB(nn.Module):
    def __init__(self, image_size):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER_SIZE)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(SECOND_HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelC(nn.Module):
    def __init__(self, image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER_SIZE)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(SECOND_HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelD(nn.Module):
    def __init__(self, image_size):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER_SIZE)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(SECOND_HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
        self.b1 = nn.BatchNorm1d(FIRST_HIDDEN_LAYER_SIZE)
        self.b2 = nn.BatchNorm1d(SECOND_HIDDEN_LAYER_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.b1(self.fc0(x)))
        x = F.relu(self.b2(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelE(nn.Module):
    def __init__(self, image_size):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, OUTPUT_LAYER_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


class ModelF(nn.Module):
    def __init__(self, image_size):
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, OUTPUT_LAYER_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


class MyDataSet(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        # get number of instances and labels
        return len(self.data)

    def __getitem__(self, idx):
        # return instance and it's label
        return self.data[idx], self.labels[idx].type(torch.LongTensor)


def init_load(batch_size):
    # transforms_norm = tr.Compose([tr.ToTensor(), tr.Normalize((0.5,), (0.5,))])
    # need to check if need to normalize or not, checked
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=tr.ToTensor(), download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=tr.ToTensor())

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(num_train * 0.2)

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, sampler=validation_sampler)
    # for Q3
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    transform = tr.Compose([tr.ToTensor()])
    test_loader = np.loadtxt(sys.argv[3])
    test_loader /= 255.0
    test_loader = transform(test_loader)[0].float()

    return train_loader, test_loader, validation_loader


def train(model, train_loader, optimizer, optimizerB):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        flag = 0
        if isinstance(model, (ModelB, ModelE, ModelF)):
            flag = 1
        if flag == 1:
            optimizerB.zero_grad()
        else:
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        if flag == 1:
            optimizerB.step()
        else:
            optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        train_loss += loss
        if flag == 1:
            optimizerB.step()
        else:
            optimizer.step()
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    train_loss /= (len(train_loader))
    acc = ((100. * correct.item()) / (len(train_loader) * BATCH_SIZE))
    return train_loss, acc


# test the model
def validate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader)
    # Printing helper for us
    #print('\nTest Set: Avg Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    acc = (100. * correct.item()) / (len(test_loader))
    return test_loss, acc


def test(model, testx):
    file = open("test_y", 'w+')
    for i in testx:
        out = model(i)
        pred = out.max(1, keepdim=True)[1].item()
        mwss = str(pred) + '\n'
        file.write(mwss)
    file.close()


def train_validation_graphs(avg_train_loss, avg_validation_loss):
    line1, = plt.plot(list(avg_train_loss.keys()), list(avg_train_loss.values()), "blue",
                      label='Train average Loss')
    line2, = plt.plot(list(avg_validation_loss.keys()), list(avg_validation_loss.values()), "red",
                      label='Validation average Loss')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4), line2: HandlerLine2D(numpoints=4)})
    plt.show()


def acc_graphs(avg_acc_train, avg_acc_validation):
    line1, = plt.plot(list(avg_acc_train.keys()), list(avg_acc_train.values()), "blue",
                      label='Train average Accuracy')
    line2, = plt.plot(list(avg_acc_validation.keys()), list(avg_acc_validation.values()), "red",
                      label='Validation average Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4), line2: HandlerLine2D(numpoints=4)})
    plt.show()


def start_program(model, train_loader, optimizer, optimizerB, validation_loader, test_loader):
    avg_train_loss = {}
    avg_validation_loss = {}
    avg_acc_train = {}
    avg_acc_validation = {}
    for epoch in range(EPOCHS):
        avg_train_loss[epoch], avg_acc_train[epoch] = train(model, train_loader, optimizer, optimizerB)
        avg_validation_loss[epoch], avg_acc_validation[epoch] = validate(model, validation_loader)
    #train_validation_graphs(avg_train_loss, avg_validation_loss)
    #acc_graphs(avg_acc_train, avg_acc_validation)
    test(model, test_loader)


def read_data_files(train_x, train_y, test_x):
    train_x = np.loadtxt(train_x).astype(np.float32)
    train_y = np.loadtxt(train_y).astype(np.long)
    train_x = np.divide(train_x, 255.0)

    tensor_x = torch.Tensor(train_x).type(torch.float32)  # transform to torch tensor
    tensor_y = torch.Tensor(train_y).type(torch.LongTensor)

    my_dataloader = MyDataSet(tensor_x, tensor_y)

    num_train = my_dataloader.__len__()
    indices = list(range(num_train))
    split = int(num_train * 0.2)

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=my_dataloader, batch_size=BATCH_SIZE, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=my_dataloader, batch_size=1, sampler=validation_sampler)

    transform = tr.Compose([tr.ToTensor()])
    test_x = np.loadtxt(test_x)
    test_x = np.divide(test_x, 255.0)
    test_x = transform(test_x)[0].float()
    return train_loader, test_x, validation_loader



def main():
    train_loader, test_loader, validation_loader = read_data_files(sys.argv[1], sys.argv[2], sys.argv[3])
    #train_loader, test_loader, validation_loader = init_load(BATCH_SIZE)
    model = ModelB(image_size=IMAGE_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=LEARN_RATE)
    optimizerB = optim.Adam(model.parameters(), lr=LEARN_RATE)
    start_program(model=model, train_loader=train_loader, optimizer=optimizer, optimizerB=optimizerB,
                  validation_loader=validation_loader, test_loader=test_loader)


if __name__ == "__main__":
    main()
