import numpy as np
# Set data
#extract from sensor1 to heartBeat
#and delete first row
signal_data = np.loadtxt('csv/signal_after1.csv',
                         delimiter=',',
                         dtype=str)
signal_data = signal_data[:, 3:11]
signal_data = np.delete(signal_data, 0, 0)
row, col = signal_data.shape
#f:sample rate 
#term:class determined terminal
signal_dim = 8
f = 16
term = 30
input_dim = f*term*signal_dim
div, dived = divmod(row, input_dim)
#training set shape is (2160, 480)
signal_data = signal_data[:div*input_dim].reshape(-1,input_dim)
xtrain = signal_data[::2].astype(np.float32)
xtest = signal_data[1::2].astype(np.float32)
y_true = np.loadtxt('csv/SleepStage1.csv',
                     delimiter=',',
                     dtype='str')
y_true = np.delete(y_true, 0, 0)
y_true = y_true[:,1]
#All x row size is 264
y_true = y_true[:264]
ytrain = y_true[::2]
ytest = y_true[1::2]

onehot_train = np.zeros(132*6).reshape(132, 6)
onehot_test = np.zeros(132*6).reshape(132, 6)
ytrain_num = np.zeros(132)
#W = 0, R = 1, N1 = 2, N2 = 3 <= in my definitioin
#n4 = 0, n3 = 1, n2 = 2, n1 = 3, r = 4, w = 5 <= in excel

for i, j in enumerate(ytrain):
    if ytrain[i] == 'W':
        onehot_train[i][0] = 1
    if ytrain[i] == 'R':
        onehot_train[i][1] = 1
    if ytrain[i] == 'N1':
        onehot_train[i][2] = 1
    if ytrain[i] == 'N2':
        onehot_train[i][3] = 1

for i, j in enumerate(ytest):
    if ytest[i] == 'W':
        onehot_test[i][0] = 1
    if ytest[i] == 'R':
        onehot_test[i][1] = 1
    if ytest[i] == 'N1':
        onehot_test[i][2] = 1
    if ytest[i] == 'N2':
        onehot_test[i][3] = 1

#Be carafull that i should arange 0 to 5
#don't code like this
#for i in array
#Then i become the element of array. Never mean iteration!!!!
onehot_train = onehot_train.astype(int)
for num, array in enumerate(onehot_train):
    for i in range(len(array)):
        if array[i] == 1:
            ytrain_num[num] = i

onehot_train = onehot_train.astype(np.float32)
onehot_test = onehot_test.astype(np.float32)