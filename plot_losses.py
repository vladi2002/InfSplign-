import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Input range
x = torch.linspace(-1, 1, 500)

# Parameters (used in your code)
alpha = 5.0
margin = 0.25

# Loss functions
sigmoid_loss = torch.sigmoid(x) # alpha *
relu_loss = F.relu(x - margin)
# squared_relu_loss = F.relu(margin - x) ** 2
gelu_loss = F.gelu(x - margin)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), sigmoid_loss.numpy(), label="Sigmoid") # : σ(x)
plt.plot(x.numpy(), relu_loss.numpy(), label="ReLU") # : ReLU(m - x)
# plt.plot(x.numpy(), squared_relu_loss.numpy(), label="Squared ReLU: ReLU(m - x)²")
plt.plot(x.numpy(), gelu_loss.numpy(), label="GELU") # : GELU(m - x)

plt.title("Comparison of Spatial Loss Functions")
plt.xlabel("x = object distance")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
