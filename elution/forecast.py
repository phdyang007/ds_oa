#using LSTM as forecasting engine

import pandas as pd 
import numpy as np  

df = pd.read_excel("data_daily.xlsx")

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)

#split training and testing


x_df = df.iloc[:,3:]
y_df = df.iloc[:,1:3]
x_train_df = x_df[0:int(n*0.8)]
x_test_df = x_df[int(n*0.8):]
y_train_df = y_df[0:int(n*0.8)]
y_test_df = y_df[int(n*0.8):]



print("Training Shape", x_train_df.shape, y_train_df.shape)
print("Testing Shape", x_test_df.shape, y_test_df.shape) 

import torch 
import torch.nn as nn
from torch.autograd import Variable 

x_train_tensors = Variable(torch.Tensor(x_train_df.values))
x_test_tensors = Variable(torch.Tensor(x_test_df.values))

y_train_tensors = Variable(torch.Tensor(y_train_df.values))
y_test_tensors = Variable(torch.Tensor(y_test_df.values)) 

print(x_train_tensors.shape, y_train_tensors.shape)
x_train_tensors_final = torch.reshape(x_train_tensors,(x_train_tensors.shape[0],1, x_train_tensors.shape[1]))
x_test_tensors_final = torch.reshape(x_test_tensors,(x_test_tensors.shape[0],1,x_test_tensors.shape[1])) 


class mylstm(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, seq_length):
        super(mylstm, self).__init__()
        self.num_classes = output_size #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, output_size) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out


num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 8 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 2 #number of output classes 


lstm1 = mylstm(num_classes, input_size, hidden_size, num_layers, x_train_tensors_final.shape[1]) #our lstm class