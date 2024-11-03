**Wine Quality Classification using Single Neuron Model**

This project aims to classify red and white wines based on their chemical properties using a single neuron model. The classification is performed using a dataset from the UCI Machine Learning Repository.

**Required Libraries**
Before running the code, make sure you have the following libraries installed:
•	pandas
•	numpy
•	matplotlib
You can install these libraries using pip:
bash
pip install pandas numpy matplotlib

**Project Structure**
1.	Data Loading and Preprocessing: The code reads red and white wine datasets, assigns labels to distinguish between them, combines and shuffles the data.
2.	Feature Extraction: Extracts specific features for the classification task.
3.	Model Definition: Defines a single neuron model to classify the wines.
4.	Training the Model: Trains the model using negative log-likelihood loss.
5.	Evaluation: Evaluates the model's classification accuracy.
6.	Visualization: Plots the dataset and decision boundary in 3D.

**Usage**
**Data Loading**
The wine quality datasets are loaded into pandas DataFrames, with red wine assigned a label of 1 and white wine assigned a label of 0.

**Feature Extraction**
The following columns are extracted for classification:
•	citric acid
•	residual sugar
•	total sulfur dioxide

**Model Definition**
A single neuron model is defined with methods for forward propagation, activation, gradient calculation, and weight updates.
**Training**
The model is trained using the negative log-likelihood loss function over a specified number of epochs. The learning rate and number of epochs can be adjusted as needed.
**Evaluation**
The model's performance is evaluated by calculating the classification accuracy.
**Visualization**
The dataset and decision boundary are plotted in 3D for better understanding and visualization of the classification results.

**Conclusion**
This project demonstrates how to classify wine quality based on chemical properties using a simple neural network model. Feel free to experiment with different features, learning rates, and epochs to improve the model's performance.

