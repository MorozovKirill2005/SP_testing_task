from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import optim

#функция-обучения нейронной сети
def train_loop(data, model, func_loss, optimizer):
    size = len(data.dataset)

    model.train()
    running_loss = 0.0
    for batch, (inputs, labels) in enumerate(data, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = func_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch + 1) % 2000 == 0:
            print(f"loss: {running_loss / 2000:>7f}  [{batch + 1:>5d}/{size:>5d}]")
            running_loss = 0.0

#функция-тестирование нейронной сети
def test_loop(data, model, func_loss):
    correct = 0
    total = 0
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for (inputs, labels) in data:
            outputs = model(inputs)
            test_loss = func_loss(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%, loss: %.7f' % (100 * correct / total, test_loss / (total / batch_size)))
    

#выбор гиперпараметров
batch_size = 4
learn_rate = 1e-3
epoch = 10
momentum = 0.9

#Преобразование данных: перевод в тензор из PyTorch и нормализация
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#загрузка датасета
data_train = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
data_test = datasets.CIFAR10(root = "./data", train = False, download = False, transform = transform)
#разбитие данных на партии
dataload_train = DataLoader(data_train, batch_size = batch_size, shuffle = True)
dataload_test = DataLoader(data_test, batch_size = batch_size, shuffle = True)

#представление модели нейронной сети
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

#инициализация модели свёрточной нейронной сети, выбор функции потерь и оптимизитора
model = NeuralNetworks()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learn_rate, momentum = momentum)

#цикл обучения и проверки результатов обучения нейронной сети
for ep in range(epoch):
    print(f"Epoch {ep + 1}\n-------------------------------")
    train_loop(dataload_train, model, loss_func, optimizer)
    test_loop(dataload_test, model, loss_func)

#сохранение модели
torch.save(model, 'model.pth')