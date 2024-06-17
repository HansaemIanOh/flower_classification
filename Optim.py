import os
from collections import Counter
from Model import *
from Model2 import *
import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import copy
import time
from torchvision.datasets import MNIST

class Optim:
    def __init__(self, device, lr, num_classes, epochs, batch_size, model_name):
        self.device = device

        self.lr = lr
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = model_name
        # self.model = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=num_classes, heads=16, input_resolution=224, width=64)
        if model_name == 'ResNet18':
            self.model = ResNet18(num_classes)
        elif model_name == 'ResNet34':
            self.model = ResNet34(num_classes)
        elif model_name == 'ResNet50':
            self.model = ResNet50(num_classes)
        elif model_name == 'ModifiedResNet':
            self.model = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=num_classes, heads=16, input_resolution=224, width=64)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.model = self.model.to(self.device)
        # summary(self.model, input_size=(3, 224, 224)) # gpu 0
        # exit()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor()
        ])
    def Train(self, train_path, model=None):
        def N2T(data): 
            # data = np.moveaxis(data, -1, 1)
            data = data.cpu().detach().numpy()
            return torch.tensor(data / 255.0, device=self.device, dtype=torch.float32)
        def D2H(data): 
            data = data.cpu().detach().numpy()
            return torch.nn.functional.one_hot(torch.tensor(data, device=self.device), num_classes=self.num_classes)
        
        # ==================== 
        dataset = ImageFolder(train_path, transform=self.data_transforms)
        # # 라벨 목록 가져오기
        # class_names = dataset.classes

        # # 각 데이터의 라벨 개수 집계
        # label_counts = Counter([label for _, label in dataset])

        # # 결과 출력
        # for class_name, count in zip(class_names, [label_counts[i] for i in range(len(class_names))]):
        #     print(f'Class: {class_name}, Count: {count}')
        # '''
        # Class: daisy, Count: 633
        # Class: dandelion, Count: 898
        # Class: roses, Count: 641
        # Class: sunflowers, Count: 699
        # Class: tulips, Count: 799
        # '''
        # exit()
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        train_indices, valid_indices = train_test_split(indices, test_size=0.2, random_state=42)
        train_subset = Subset(dataset, train_indices)
        valid_subset = Subset(dataset, valid_indices)
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        # DataLoader에서 배치를 순회하며 라벨 확인
        # ====================
        # for batch_idx, (x, y) in enumerate(train_loader):
        #     print(f'Batch {batch_idx + 1}')
        #     unique_labels = y.unique()
        #     for label in unique_labels:
        #         print(f'Label index: {label.item()}, Class name: {class_names[label.item()]}')
        #     print('---')
        # exit()
        # ====================
        valid_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        # ====================
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad='amsgrad')
        if model is not None:
            self.model = model
            print("Loaded pre-trained model. Continuing training.")
        else:
            print("Training from scratch.")
        self.model.train()
        best_model = copy.deepcopy(self.model)
        history = []
        print("===== Training Start =====")
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            losses = []
            accuracies = []
            for x, y in train_loader:
                Tx = N2T(x)
                Ty = D2H(y)
                optimizer.zero_grad()
                loss, accuracy = self.Loss(Tx, Ty, self.model)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            accuracies.append(accuracy)
            losses = np.array(losses)
            avg_loss = np.mean(losses)
            avg_accuracy = np.mean(np.array(accuracies))
            with torch.no_grad():
                valid_accuracies = 0
                valid_accuracies_c = 0
                N = 0
                for x, y in valid_loader:
                    Vx = N2T(x)
                    Vy = D2H(y)
                    loss_valid, valid_accuracy = self.Loss(Vx, Vy, self.model)
                    loss_valid_comparison, valid_accuracy_c = self.Loss(Vx, Vy, best_model)
                    valid_accuracies += valid_accuracy
                    valid_accuracies_c += valid_accuracy_c
                    
                    N += 1

                valid_accuracies = valid_accuracies / N
                valid_accuracies_c = valid_accuracies_c / N
                if valid_accuracies > valid_accuracies_c:
                    best_model = copy.deepcopy(self.model)
                    print("Model stored.")
                    self.Save(best_model)

            epoch_train_time = time.time() - epoch_start_time
            print('Epoch : {} / {} Time : {:.3f} Loss : {:.5f} Accuracy : {:.5f} Valid Accuracy : {:.5f}'.format(epoch + 1, self.epochs, epoch_train_time, avg_loss, avg_accuracy, valid_accuracies))
            # loss evolution
            loss_save = [avg_loss, loss_valid.item()]
            history.append(loss_save)
        return best_model, history
    def Loss(self, x, y, model):
        y_hat = model(x)
        loss = F.cross_entropy(y_hat.float(), y.float())
        # ==================== accuracy 
        _, predicted_labels = torch.max(y_hat, 1)
        true_labels = torch.argmax(y, dim=1)
        correct_predictions = (predicted_labels == true_labels).sum().item()
        accuracy = correct_predictions / y.size(0)
        # ====================
        return loss, accuracy
    def Save(self, model):
        os.makedirs('Parameters/'+'resnet', exist_ok=True)
        save = os.path.join('Parameters/'+'resnet', self.model_name+".pth")
        torch.save(model.state_dict(), save)
