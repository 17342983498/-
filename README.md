# -import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
df = pd.read_csv('D:/2022-3-21/dataaset/train.csv', header=None, encoding='big5')
print(df.shape)
df.head()
df = pd.read_csv('D:/2022-3-21/dataaset/X_train', header=None, encoding='big5')
print(df.shape)
df.head()
np.random.seed(0)
X_train_fpath = 'D:/2022-3-21/dataaset/X_train'
Y_train_fpath = 'D:/2022-3-21/dataaset/Y_train'
X_test_fpath = 'D:/2022-3-21/dataaset/X_test'
output_fpath = 'D:/2022-3-21/dataaset/output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)


# 规范化
def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)

    return X, X_mean, X_std


def _train_dev_split(X, Y, dev_ratio=0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

# Split data into training set and development set
dev_ratio = 0.1
# 9:1
X_train, Y_train, X_eval, Y_eval = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
eval_size = X_eval.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

print('Size of training set: {}'.format(X_train.shape[0]))
print('Size of eval set: {}'.format(X_eval.shape[0]))
print('Size of testing set: {}'.format(X_test.shape[0]))
print('Dimension of data: {}'.format(data_dim))


def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)


def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)


def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc
# 交叉熵损失 if 0-1损失 第二项为0
def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

#
def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


# Zero initialization for weights ans bias
w = np.zeros((data_dim,))
b = np.zeros((1,))

# Some parameters for training
EPOCH = 10
batch_size = 128
learning_rate = 0.2

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
eval_loss = []
train_acc = []
eval_acc = []

# Calcuate the number of parameter updates
adagrad_step = 1
batch_step = 0
# Iterative training
for epoch in range(EPOCH):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)

    # Mini-batch training
    step = 0
    steps = int(np.floor(train_size / batch_size))
    for idx in range(steps):  # floor(48830/128)=382
        X = X_train[idx * batch_size:(idx + 1) * batch_size]
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b)

        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate / np.sqrt(adagrad_step) * w_grad
        b = b - learning_rate / np.sqrt(adagrad_step) * b_grad

        step += 1
        adagrad_step += 1

        # Compute loss and accuracy of training set and development set
        y_train_pred = _f(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        y_eval_pred = _f(X_eval, w, b)
        Y_eval_pred = np.round(y_eval_pred)

        acc_train = _accuracy(Y_train_pred, Y_train)
        loss_train = _cross_entropy_loss(y_train_pred, Y_train) / train_size
        acc_eval = _accuracy(Y_eval_pred, Y_eval)
        loss_eval = _cross_entropy_loss(y_eval_pred, Y_eval) / eval_size

        if step % 50 == 0 or step == steps:
            print(
                f'Epoch {epoch}/{EPOCH}, step {step}/{steps} : train_loss = {loss_train}, train_acc = {acc_train}, eval_loss = {loss_eval}, eval_acc = {acc_eval}')

        train_acc.append(acc_train)
        train_loss.append(loss_train)
        eval_acc.append(acc_eval)
        eval_loss.append(loss_eval)

print('Training loss: {}'.format(train_loss[-1]))
print('Eval loss: {}'.format(eval_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Eval accuracy: {}'.format(eval_acc[-1]))
