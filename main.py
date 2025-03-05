from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Hyperparametrs
batch_size = 4

#Преобразование данных: перевод в тензор из PyTorch и нормализация
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#загрузка датасета
data_train = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
data_test = datasets.CIFAR10(root = "./data", train = False, download = False, transform = transform)
#разбитие данных на партии
dataload_train = DataLoader(data_train, batch_size = batch_size, shuffle = True)
dataload_test = DataLoader(data_test, batch_size = batch_size, shuffle = True)