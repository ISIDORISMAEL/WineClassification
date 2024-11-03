# These are the packages required for this assignment
import pandas as pd
import numpy as np
import math, random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent line-wrapping
pd.set_option('display.max_rows', None)  # Show all rows

# Use Pandas to read the csv file into a dataframe.
# Note that the delimiter in this csv is the semicolon ";" instead of a ,
df_red = pd.read_csv('winequality-red.csv',delimiter=";")

# Because we are performing a classification task, we will assign all red wine a label of 1
df_red["color"] = 1

# The method .head() is super useful for seeing a preview of our data!
# df_red.head()
print(df_red.head())


df_white = pd.read_csv('winequality-white.csv',delimiter=";")
df_white["color"] = 0  #assign white wine the label 0
# df_white.head()
print(df_white.head())

# Now we combine our two dataframes
df = pd.concat([df_red, df_white])

# And shuffle them in place to mix the red and white wine data together
df = df.sample(frac=1).reset_index(drop=True)
# df.head()
print(df.head())

# We choose three attributes of the wine to perform our prediction on
input_columns = ["citric acid", "residual sugar", "total sulfur dioxide"]
output_columns = ["color"]

# We extract the relevant features into our X and Y numpy arrays
X = df[input_columns].to_numpy()
Y = df[output_columns].to_numpy()
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
in_features = X.shape[1]

# Your Code Here!
class singleNeuronModel():
    def __init__(self, in_features):
        self.w = 0.01 * np.random.randn(in_features)
        self.w_0 = 0.01 * np.random.randn()
        self.non_zero_tolerance = 1e-8

    def forward(self, X):
        self.z = X @ self.w.T + self.w_0
        self.a = self.activation(self.z)
        return self.a

    # classification_model = ...
    def activation(self, z):
        return 1 / (1 + np.exp(-z) + self.non_zero_tolerance)

    def gradient(self, x):
        self.grad_w = self.a * (1 - self.a) * x
        self.grad_w_0 = self.a * (1 - self.a)

    def update(self, grad_loss, learning_rate):
        model.w -= grad_loss * learning_rate * self.grad_w * learning_rate
        model.w_0 -= grad_loss * learning_rate * self.grad_w_0 * learning_rate


def train_model_NLL_loss(model, input_data, output_data, learning_rate, num_epochs):
    non_zero_tolerance = 1e-8
    num_samples = len(input_data)
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.

        for i in range(num_samples):
            x = input_data[i, ...]
            y = output_data[i]
            y_predicted = model.forward(x)

            loss = -(y * np.log(y_predicted + non_zero_tolerance) + (1 - y) * np.log(
                1 - y_predicted + non_zero_tolerance))
            total_loss += loss

            model.gradient(x)

            grad_loss = (y_predicted - y) / (y_predicted * (1 - y_predicted))
            model.update(grad_loss, learning_rate)
        report_every = max(1, num_epochs // 10)
        if epoch == 1 or epoch % report_every == 0:
            print("Epoch:", epoch, "Loss:", total_loss)


# train the model...
learning_rate = 0.001
epochs = 200


def plot_3d_dataset(x, y):
    x_np_3d = np.array(x)
    x_np_3d.reshape(len(x), 3)
    colors = []
    for label in y:
        if label == 1:
            colors.append('red')  # for red wine
        else:
            colors.append('green')  # for white wine
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_np_3d[:, 0], x_np_3d[:, 1], x_np_3d[:, 2], c=colors)
    plt.show()
    return ax


def plot_3d_decision_boundary(ax, w, w_0):
    X = np.linspace(-200, 200, 200)
    Y = np.linspace(-200, 200, 200)
    X, Y = np.meshgrid(X, Y)
    Z = (-X * w[0] - Y * w[1] - w_0) / w[2]
    sur = ax.plot_surface(X, Y, Z)
    ax.set_xlim(-200, 200)
    return


# We will use this function to evaluate how well our trained classifier perfom
# Hint: the model you define above must have a .forward function in order to be compatible
# Hint: this evaluation function is identical to those in previous notebooks
def evaluate_classification_accuracy(model, input_data, labels):
    # Count the number of correctly classified samples given a set of weights
    correct = 0
    num_samples = len(input_data)
    for i in range(num_samples):
        x = input_data[i, ...]
        y = labels[i]
        y_predicted = model.forward(x)
        label_predicted = 1 if y_predicted > 0.5 else 0
        if label_predicted == y:
            correct += 1
    accuracy = correct / num_samples
    print("Our model predicted", correct, "out of", num_samples,
          "correctly for", accuracy * 100, "% accuracy")
    return accuracy


# print("\nFirst set")
# input_data = np.array([[0.00, 1.9, 34.0], [0.00, 2.6, 67.0], [0.04, 2.3, 54.0], [0.56, 1.9, 60.0], [0.00, 1.9, 34.0]]) #1
# labels = [1, 1, 1, 1, 1] #1

# model = SingleNeuronClassificationModel(in_features=len(input_data[0]))
# train_model_NLL_loss(model, input_data, labels, learning_rate=0.01, num_epochs=200)

# print("w", model.w)
# print("w_0", model.w_0)

# evaluate_classification_accuracy(model, input_data, labels)

# print("\nEnd First set")

# print("\nSecond set")
# input_data = np.array([[0.36, 20.7, 170.0], [0.34, 1.6, 132.0], [0.40, 6.9, 97.0], [0.32, 8.5, 186.0], [0.32, 8.5, 186.0]]) #2
# labels = [1, 0, 0, 1, 1] #2

# model = SingleNeuronClassificationModel(in_features=len(input_data[0]))
# train_model_NLL_loss(model, input_data, labels, learning_rate=0.01, num_epochs=200)

# print("w", model.w)
# print("w_0", model.w_0)

# evaluate_classification_accuracy(model, input_data, labels)

# print("\nEnd Second set")

print("\nThird set")
# labels = [0, 0, 0, 1, 0]  # 3
# input_data = np.array(
#     [[0.32, 10.5, 105.0], [0.22, 1.4, 149.0], [0.32, 1.6, 150.0], [0.32, 1.8, 8.0], [0.29, 13.7, 134.0]])  # 3

labels = df['color'].tolist()
print(labels)

extracted_data = df[input_columns]
input_data = extracted_data.to_numpy()
print(input_data)


model = singleNeuronModel(in_features=len(input_data[0]))
train_model_NLL_loss(model, input_data, labels, learning_rate=0.01, num_epochs=200)

print("\nEnd Third set")

print("Final features")

print("w", model.w)
print("w_0", model.w_0)

evaluate_classification_accuracy(model, input_data, labels)

ax = plot_3d_dataset(input_data, labels)
plot_3d_decision_boundary(ax, model.w, model.w_0)
