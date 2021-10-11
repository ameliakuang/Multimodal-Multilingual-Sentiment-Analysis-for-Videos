import torch
from torch import nn
import numpy as np

class MLP_net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MLP_net, self).__init__()

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.output_size, 1)
        )
        
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = torch.squeeze(x,1)
        return x


class MLP:
    def __init__(self,input_size=17,hidden_size=50,output_size=7,epochs=10):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.network = MLP_net(self.input_size,self.hidden_size,self.output_size)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.epochs=epochs

    def train(self,train_loader,valid_loader):
        mean_train_losses = []
        mean_valid_losses = []
        epoch_list=[]

        for epoch in range(self.epochs):
            #print("epoch:",epoch+1)
            self.network.train()
            
            train_losses = []
            for i, (features, labels) in enumerate(train_loader):
                features=features.float()
                labels=labels.float()
                
                self.optimizer.zero_grad()
                
                outputs = self.network(features)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())                
                # if i%2 == 0:
                #     print(i '/ 50000')

            if (epoch+1)%50==0 or epoch==0:
                valid_losses=self.test(valid_loader)

                epoch_list.append(epoch)
                mean_train_losses.append(np.mean(train_losses))
                mean_valid_losses.append(np.mean(valid_losses)) 

                print('Validation... epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(valid_losses)))
        return mean_train_losses, mean_valid_losses, epoch_list
    

    def test(self,valid_loader):
        self.network.eval()

        valid_acc_list = []
        valid_losses = []

        with torch.no_grad():
            for i, (features, labels) in enumerate(valid_loader):
                features=features.float()
                labels=labels.float()

                outputs = self.network(features)
                loss = self.loss_fn(outputs, labels)
                valid_losses.append(loss.item())

        return valid_losses

    # def test(self,valid_loader):
    #     self.network.eval()

    #     valid_acc_list = []
    #     valid_losses = []

    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for i, (features, labels) in enumerate(valid_loader):
    #             features=features.float()
    #             labels=labels.float()

    #             outputs = self.network(features)
    #             print("outputs:",outputs.data)
    #             loss = self.loss_fn(outputs, labels)
    #             print("loss:",loss)
    #             valid_losses.append(loss.item())
                
    #             _, predicted = torch.max(outputs.data, 1)
    #             correct += (predicted == labels).sum().item()
    #             total += labels.size(0)
        
    #     accuracy = 100*correct/total
    #     valid_acc_list.append(accuracy)

    #     return valid_losses, accuracy