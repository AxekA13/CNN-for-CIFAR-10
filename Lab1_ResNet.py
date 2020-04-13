from torchvision.models import resnet50
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


class CIFAR_ResNet(nn.Module):
    def __init__(self):
        super(CIFAR_ResNet, self).__init__()
        resnet = resnet50(pretrained=True)
        layers = list(resnet.children())[:-1]  # delete classifier

        self.resnet = nn.Sequential(*layers)
        self.fc1 = nn.Linear(resnet.fc.in_features, 256)
        self.BN1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.BN2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
            x = x.view(-1, x.size(0))

        x = self.BN1(self.fc1(x))
        x = F.relu(x)
        x = self.BN2(self.fc2(x))
        x = F.relu(x)
        # x = F.dropout(x,p=0.4, training=self.training)
        x = self.fc3(x)
        return x


def train(epoch, model):
    CNN = model
    train_accuracy = 0
    train_loss = 0
    total = 0
    N_count = 0
    for batch, (X, label) in enumerate(train_data):
        X, label = X.to(device), label.to(device)

        N_count += X.size(0)
        optimizer.zero_grad()

        output = CNN(X)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        train_accuracy = accuracy_score(label.cpu().data.squeeze().numpy(), predicted.cpu().data.squeeze().numpy())

    print('Train Epoch: {} [{}/{}] \tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
        epoch + 1, N_count, len(train_data.dataset), train_loss / len(train_data),
        100 * (train_accuracy)))

    return 100 * train_accuracy, train_loss / N_count


def test(epoch, model):
    CNN = model
    CNN.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, label in test_data:
            X, label = X.to(device), label.to(device)

            output = CNN(X)
            loss = criterion(output, label)
            test_loss += loss.item()

            _, pred = torch.max(output, 1)
            acc = accuracy_score(label.cpu().data.squeeze().numpy(), pred.cpu().data.squeeze().numpy())

    test_loss /= len(test_data)
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(test_data),
                                                                                        test_loss,
                                                                                        100 * acc))
    return test_loss / len(test_data), 100 * acc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 100
    epochs = 50

    transform = tf.Compose([tf.ToTensor(), tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = CIFAR10(root=r'C:\Users\nasty\PycharmProjects\DL\dataset\train', train=True, transform=transform,
                            download=True)
    train_data = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataset_test = CIFAR10(root=r'C:\Users\nasty\PycharmProjects\DL\dataset\test', train=False, transform=transform,
                           download=True)
    test_data = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    CNN = CIFAR_ResNet().to(device)
    print('Просто модель')
    print(CNN)
    print('Саммари модели')
    print(summary(CNN, (3, 32, 32), 1))

    if torch.cuda.device_count() > 1:
        CIFAR_ResNet = nn.DataParallel(CIFAR_ResNet)
        parameters = (list(CIFAR_ResNet.module.fc1.parameters()) + list(CIFAR_ResNet.module.BN1.parameters()) \
                      + list(CIFAR_ResNet.module.fc2.parameters()) + list(CIFAR_ResNet.module.BN2.parameters()) + \
                      list(CIFAR_ResNet.module.fc3.parameters()))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(CNN.parameters(), lr=0.0001)

    train_loss = []
    test_loss = []

    train_accuracy = []
    test_accuracy = []
    for epoch in range(epochs):
        train_acc, train_losses = train(epoch, CNN)
        test_losses, test_acc = test(epoch, CNN)
        train_accuracy.append(train_acc)
        train_loss.append(train_losses)
        test_accuracy.append(test_acc)
        test_loss.append(test_losses)

    fig, ax = plt.subplots()
    plt.title('Loss')
    plt.ylabel('Value of Loss')
    plt.xlabel('Epochs')
    plt.plot(train_loss)
    plt.legend('Train')
    plt.plot(test_loss)
    plt.legend('Test')
    fig.savefig('Loss.png')
    plt.show()
    plt.close()

    fig1, ax1 = plt.subplots()
    plt.title('Accuracy')
    plt.plot(train_accuracy)
    plt.legend('Train')
    plt.plot(test_accuracy)
    plt.legend('Test')
    plt.ylabel('Value of Accuracy')
    plt.xlabel('Epochs')
    fig1.savefig('Accuracy.png')
    plt.show()
    plt.close()

    print('Total train epochs Accuracy = ', str(sum(train_accuracy) / len(train_accuracy)))
    print('Total test epochs Accuracy = ', str(sum(test_accuracy) / len(test_accuracy)))
    print('Total train epochs Loss = ', str(sum(train_loss) /
                                            len(train_loss)))
    print('Total test epochs Loss = ', str(sum(test_loss) / len(test_loss)))
