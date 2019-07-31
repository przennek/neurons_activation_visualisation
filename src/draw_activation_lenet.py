import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

def relu(x):
    a = []
    for item in x:
        a.append(max(0, item))
    return a

def max(a, b):
    if(a > b):
        return a
    return b

def tanh(x):
    a = []
    for item in x:
        a.append(math.tanh(item))
    return a

x = np.arange(-5, 5, 0.1)
plt.grid(True)

plt.xticks(np.arange(-5, 5, 1))


# sigmoid
sig = sigmoid(x)

# relu
rel = relu(x)

#plt.plot(x, sig)
#plt.legend(['Sigmoid'])

#plt.plot(x, rel)
#plt.legend(['ReLU'])

plt.plot(x, tanh(x))
plt.legend(['tanh'])

plt.show()
