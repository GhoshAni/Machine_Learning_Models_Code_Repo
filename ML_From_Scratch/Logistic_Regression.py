# create a logistic regression using numpy
# the steps to create a logistic regression
# initialize the weights, learning rate and intercept
# predict using linear equation
# Apply sigmoid function to the predicted output
# calculate cost function 
# apply gradient descent and update the cost function at every iteration
# keep iterating untill the log loss is minimized
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

# step1 generate dummy data
ClassData = make_classification(n_features =  10, n_samples = 1000 , random_state = 4 )
x = ClassData[0]
y = ClassData[1]

print(x.shape)
print(y.shape)

# initialize weights

learning_rate = 0.01
n_iter = 100
b = 0
w = np.zeros(x.shape[1])

# Define functions

# predict = W.X_Transpose + B
predict = lambda x, w, b: np.matmul(w, x.T + b)

sigmoid = lambda yhat : 1/(1 + np.exp(-yhat))

loss = lambda y, sigmoid: - (y.np.log(sigmoid) + (1-y).np.log(1-sigmoid))

dldw = lambda x, y, sigmoid: (np.reshape(sigmoid - y, (1000,1))* x).mean(axis = 0)

dldb = lambda y, sig: (sig-y).mean(axis = 0)

update = lambda a, g, lr : a - (g* lr)
print('functions created  successfully')

for i in range(n_iter):
    yhat = predict(x,w,b)
    sig = sigmoid(yhat)
    grad_w = dldw(x,y,sig)
    grad_b = dldb(y,sig)
    w = update(w,grad_w,learning_rate)
    b = update(b,grad_b,learning_rate)
    
print('weights updated successfully')

from sklearn.metrics import classification_report
yhat = predict(x,w,b)
sigy = sigmoid(yhat)
ypred = sigy >= 0.5
print(classification_report(y,ypred))