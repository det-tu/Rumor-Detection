import torch
import numpy as np
import math
import LSTM


input_size = 10
hidden_size = 10
output_size = 10

train_num = 1000

net = LSTM.LSTM(input_size, hidden_size, output_size)
optimizer = torch.optim.SGD(net.parameters(), lr=2)
loss_func = torch.nn.MSELoss()

file_train = open('a.csv','r')
x = torch.zeros(5,10)
y = torch.zeros(10)

for t in range(train_num):
    line = file_train.readline().split(',')

    for k in range(10):
        x[0][k] = float(line[k])
    '''
        ...
    '''
    for k in range(10):
        y[k] = float(line[k])

    x_form = x[:, np.newaxis]
    y_form = y[np.newaxis, :]
    
    prediction = net(x_form)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    

file_train.close()
#torch.save(net.state_dict(),'data.pkl')
print('train complete: ',train_num)