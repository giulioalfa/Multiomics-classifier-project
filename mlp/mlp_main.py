import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os 
import numpy as np




# architecture of multi-layer perceptron architecture and forward function
class MLP(nn.Module):
    def __init__(self, n_classes, n_inputs):
        super(MLP,self).__init__()
        self.n_classes = n_classes
        self.n_inputs = n_inputs
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(n_inputs, 512)
        nn.init.xavier_normal_(self.fc1.weight)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512,512)
        nn.init.xavier_normal_(self.fc2.weight)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, n_classes)
        nn.init.xavier_normal_(self.fc3.weight)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, self.n_inputs)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.droput(x)
          # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.droput(x)
        # add output layer
        x = self.fc3(x)
        return x







def prepare_trte_data(data_folder):
    
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    view_list = [1,2,3]
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    
      

    
    
    
    
    

    data_train = torch.FloatTensor(np.concatenate((data_tr_list[0], data_tr_list[1], data_tr_list[2]), axis = 1))

    data_test = torch.FloatTensor(np.concatenate((data_te_list[0], data_te_list[1], data_te_list[2]), axis = 1))
    

    tr_labels = torch.LongTensor(np.array(labels_tr))
    te_labels = torch.LongTensor(np.array(labels_te))
    
    
    return data_train, data_test, tr_labels, te_labels




def main(data_folder):

    data_path = "Data/"
    if data_folder == 'ROSMAP' or data_folder == 'LuadLusc100':
        num_class = 2
    elif data_folder == 'BRCA' or data_folder == '5000samples':
        num_class = 5

    X_train, X_test, y_train, y_test = prepare_trte_data(data_path+data_folder)
    dim_in = X_train.shape[1]

    net = MLP(num_class, dim_in)
    loss_main = nn.CrossEntropyLoss()
    loss_second = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    num_epochs = 2500
    assert num_class != 0

    def train_net(net, x, labels, criterion, optimizer):
      ci = net(x)
      loss = criterion(ci, labels)
      loss.backward()
      optimizer.step()
      return loss.detach().cpu().numpy().item()

    def test_net(net, x, labels, criterion):
      ci = net(x)
      loss = criterion(ci, labels)
      prob = F.softmax(ci, dim=1).data.cpu().numpy()
      return ci.data.cpu().numpy()

    acc = 0
    best_epoch = 0
    for epoch in range(1, num_epochs+1):

      # Train
      optimizer.zero_grad()
      net.fc1.train(True)
      net.fc2.train(True)
      net.fc3.train(True)
      l = train_net(net, X_train, y_train, loss_main, optimizer)

      # Test
      net.fc1.train(False)
      net.fc2.train(False)
      net.fc3.train(False)
      prob = test_net(net, X_test, y_test, loss_main)
      new_acc = accuracy_score(y_test, prob.argmax(1))
      if new_acc > acc:
          acc = new_acc
          best_epoch = epoch
      if epoch % 50 == 0:
        print("\nTest: Epoch {:d}".format(epoch))
        if num_class == 2:
          print("Test ACC: {:.3f}".format(accuracy_score(y_test, prob.argmax(1))))
          print("Test F1: {:.3f}".format(f1_score(y_test, prob.argmax(1))))
          print("Test AUC: {:.3f}".format(roc_auc_score(y_test, prob[:,1])))
        else:
          print("Test ACC: {:.3f}".format(accuracy_score(y_test, prob.argmax(1))))
          print("Test F1 weighted: {:.3f}".format(f1_score(y_test, prob.argmax(1), average='weighted')))
          print("Test F1 macro: {:.3f}".format(f1_score(y_test, prob.argmax(1), average='macro')))

    print(f'Best epoch: {best_epoch} - ACC: {acc}')




if __name__ == "__main__":

    main("LuadLusc100")
