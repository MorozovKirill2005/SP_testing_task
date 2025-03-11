import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

batch_size = 4

'''def data_noise(data):
    new_data = data
    for i in range(50000):
        new_data[i] += torch.rand(data[i]) * 0.2 - 0.1
    return new_data'''

#Преобразование данных: перевод в тензор из PyTorch и нормализация
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#загрузка датасета
data_train = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
data_test = datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform)

dataload_train = DataLoader(data_train, batch_size = batch_size, shuffle = True)
#new_data = data_noise(data_train)

print(type(dataload_train))
#вывод 9 картинок
figure = plt.figure(figsize = (5, 5))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    image, labels = data_train[i]
    figure.add_subplot(rows, cols, i)
    plt.title(labels)
    plt.axis("off")
    plt.imshow(image.numpy().transpose((1, 2, 0)))
plt.show()

#вывод 9 картинок
transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1),transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
data_train = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
figure = plt.figure(figsize = (5, 5))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    image, labels = data_train[i]
    figure.add_subplot(rows, cols, i)
    plt.title(labels)
    plt.axis("off")
    plt.imshow(image.numpy().transpose((1, 2, 0)))
plt.show()

transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1),transforms.ToTensor(), transforms.Normalize((0.5), (0.5)), transforms.Lambda(lambda x: x + 0.2 * torch.rand_like(x) - 0.1)])
data_train = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
figure = plt.figure(figsize = (5, 5))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    image, labels = data_train[i]
    figure.add_subplot(rows, cols, i)
    plt.title(labels)
    plt.axis("off")
    plt.imshow(image.numpy().transpose((1, 2, 0)))
plt.show()