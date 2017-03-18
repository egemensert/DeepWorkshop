import random # to prepare the dataset
import matplotlib.pyplot as plt # for plotting endavours

def affine_forward(x, w, b):
    # this layer is the function that models the dataset
    return w * x + b

def loss_function(z, y):
    # how far are the parameters to the correct representation
    return 0.5 * (y - z) ** 2

def loss_derivative(z, y):
    # error term to update weight and biases
    return  z - y

# prepare a toy dataset
xs = [x / 100. for x in xrange(-500, 500)]
ys = [0.3 * (x / 100.) + 0.2 + 0.3 * random.random() for x in xrange(-500, 500)]

# initialize <very> random parameters
omega = 5 * random.random()
beta = 5 * random.random()

# number of iterations and the length of the step-size
num_epoch = 200
learning_rate = 1e-1

# some plotting stuff, can be ignored
plt.xlim(-3, 3)
plt.ylim(-1.5, 2)
scat1 = plt.plot(xs, ys,'g-')

# initial omega and beta
print omega, beta

# yet another printing stuff
line, = plt.plot(xs, [omega * x + beta for x in xs],'r-', lw=2)

# main loop that fits the parameters

for _ in xrange(num_epoch):
    # initialize error terms
    epoch_loss = 0.
    omega_delta = 0.
    beta_delta = 0.

    for i in xrange(len(xs)):
        # for each data entry, do
        x = xs[i]
        y = ys[i]

        # affine transformation (wx + b)
        estimation = affine_forward(x, omega, beta)

        # calculate the loss to indicate how well the classifier works
        epoch_loss += loss_function(estimation, y)

        # derivate the error term to update parameters
        dx = loss_derivative(estimation, y)

        # dw = dE * x
        omega_delta += dx * x

        # db = dE
        beta_delta += dx

    # update the parameters with their average derivatives over the dataset
    omega += -learning_rate * (omega_delta / len(xs))
    beta += -learning_rate * (beta_delta / len(xs))
    loss = epoch_loss / len(xs)

    # plotting stuff, again..
    line.set_ydata([omega * x + beta for x in xs])
    plt.draw()
    plt.pause(0.2)

    # print loss & omega & beta
    print "Loss: %.4f, Omega: %.4f, Beta: %.4f " % (loss, omega, beta)
