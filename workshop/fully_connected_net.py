import numpy as np
import random

from layers import *
from optim import *

class TwoLayerNet(object):

    def __init__(self, input_dim=8*8, hidden_dim=100, num_classes=10,
                weight_scale = 1e-3, reg= 0.0, lr=1e-2, batch_size=64):

        self.params = {}
        self.reg = reg
        self.batch_size = batch_size
        self.lr = lr
        ################################################################
        # TODO: Initalize weights and biases with correct              #
        # dimensionality and scale the weights with the given value    #
        # HINT: To generate a zero vector of shape (N, ), you can use  #
        # np.zeros(N) and to generate a random matrix, of shape (N, D) #
        # np.random.rand(N, D) and for a random matrix with normal     #
        # distribution np.random.randn(N, D)                           #
        ################################################################

        ################################################################
        # End of the code                                              #
        ################################################################

    def train(self, input, target, num_epochs= 10, verbose=True, test=None):
        batch_depot = self._sample_minibatch(input, target)
        for epoch in xrange(num_epochs):
            loss = 0.
            scores = None
            for i, (X, y) in enumerate(batch_depot):
                scores, b_loss = self.train_batch(X, y, test = None)
                loss += b_loss
                if i + 1 == len(input) / self.batch_size:
                    break
            if test:
                X_t, y_t = test
                tscores, tloss = self.train_batch(X_t, y_t, test=test)
                t_accuracy = self._get_accuracy(tscores, y_t)
                test_metrics = (tloss, t_accuracy)

            if verbose:
                accuracy = self._get_accuracy(scores, y)

                self._print_training(epoch, loss, accuracy, test=test_metrics)

    def train_batch(self, X, y, test =None):
        grads = {"W1":None, "W2":None, "b1":None, "b2":None}
        ################################################################
        # TODO: Forward the given input, calculate the loss,           #
        # the scores and the error w.r.t. the label                    #
        ################################################################

        ################################################################
        # End of the code                                              #
        ################################################################
        if test:
            return scores, loss

        ################################################################
        # TODO: Backpropagate the Error                                #
        ################################################################

        ################################################################
        # End of the code                                              #
        ################################################################
        for param in grads:
            self.params[param], _ = sgd(self.params[param], grads[param],
                config={"learning_rate":self.lr})

        return scores, loss

    def _sample_minibatch(self, X, y):
        data = zip(X, y)
        random.shuffle(data)
        X, y = zip(*data)
        while True:
            for i in xrange(0, len(X) / self.batch_size, self.batch_size):
                to = i + self.batch_size if i+self.batch_size < len(X) else len(X)
                mb_inputs = X[i: to]
                mb_targets = y[i: to]
                yield np.array(mb_inputs), np.array(mb_targets)

    def _print_training(self, epoch, loss, acc, test=None):
        epoch_number = epoch + 1
        line = "Epoch {%s} | loss: %.4f, accuracy: %.4f" % (epoch, loss, acc)
        if test:
            tl, ta = test
            line += " test_loss: %.4f, test_accuracy: %.4f" % (tl, ta)
        print line

    def _get_accuracy(self, scores, y):
        scores = np.argmax(scores, axis=1)
        y = np.argmax(y, axis= 1)
        trues = np.equal(scores, y)
        accuracy = trues.sum() * 1. / len(y)
        return accuracy
