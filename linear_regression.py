# This is the first PyTorch program
# I created with a simple example of creating a linear regression model using PyTorch.
# This program was created with the help of ChatGPT

# First import the necessary libraries: PyTorch, NumPy, 
#  and the nn module from PyTorch. 

import torch
import torch.nn as nn
import numpy as np

# Define the model
# Define the LinearRegression class, which extends the nn.Module class in PyTorch.

# Inside the LinearRegression class, we define the constructor (init) and
#  the forward method, which performs the forward pass of the model.
# The constructor takes in the input size and output size of the model and
#  creates a Linear layer with those dimensions. 
# The forward method takes in an input tensor x and
#  passes it through the Linear layer to get the output.
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

# Generate some random data and convert it to PyTorch tensors. 
# We also define the model, loss function, and optimizer.

# Generate some random data
input_size = 1
output_size = 1
num_examples = 100
x = np.random.rand(num_examples, input_size)
y = 2 * x + 1

# Convert data to PyTorch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Define the model and loss function
model = LinearRegression(input_size, output_size)
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# We then train the model for a specified number of epochs, 
# performing a forward pass, calculating the loss, performing a backward pass, and
# optimizing the parameters using the SGD optimizer.

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the progress
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the model

# Test the model by passing in a test input and printing the predicted output.
with torch.no_grad():
    inputs = torch.tensor([[2.0]])
    predicted = model(inputs).detach().numpy()
    print('Predicted value: {:.2f}'.format(predicted[0][0]))