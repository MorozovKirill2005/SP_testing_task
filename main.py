from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

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

class NeuralNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120,),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    
    def forward(self, input):
        output = self.network(input)
        return output

model = NeuralNetworks()
print(model)