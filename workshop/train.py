from fully_connected_net import TwoLayerNet

from sklearn import datasets
import numpy as np

def categorical(x):
    z = np.zeros(10)
    z[x] = 1
    return z

digits, labels = datasets.load_digits(return_X_y=True)

digits /= 255.
digits -= digits.mean()
labels = np.array(map(categorical,list(labels)))
tr_split = int(len(digits) * 0.8)

tr_X = digits[:tr_split]
tr_y = labels[:tr_split]

te_X = digits[tr_split:]
te_y = labels[tr_split:]

nn = TwoLayerNet(lr=1e-2)
print len(tr_X)
#assert False
nn.train(tr_X, tr_y, test=(te_X, te_y), num_epochs = 100)
