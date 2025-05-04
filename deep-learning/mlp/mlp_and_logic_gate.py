'''
MLP for AND Logic Gate Classification

This script implements a simple Multilayer Perceptron (MLP) to solve the AND logic gate problem.
The MLP model consists of one hidden layer and uses the sigmoid activation function for both 
the hidden layer and output layer. The model is trained using the backpropagation algorithm 
and gradient descent to optimize the weights and biases.

Problem Description:
- The AND gate logic takes two binary inputs and produces a binary output.
- The task is to train the model to predict the output of an AND gate for given inputs.
- The input dataset consists of 4 samples, where each sample has two inputs (x1, x2) and 
  a corresponding output (y).

Dataset:
| x1 | x2 | Output (y) |
|----|----|------------|
| 0  | 0  | 0          |
| 0  | 1  | 0          |
| 1  | 0  | 0          |
| 1  | 1  | 1          |

Model Architecture:
- Input Layer: 2 neurons (for the two inputs x1 and x2)
- Hidden Layer: 4 neurons
- Output Layer: 1 neuron (for the output y)

Training:
- The model is trained using the Mean Squared Error (MSE) loss function.
- The backpropagation algorithm is used to update the weights and biases using gradient descent.

Expected Output:
- After training, the model should predict the correct output for all input combinations of the AND gate.

'''


import numpy as np

# Hàm kích hoạt sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Đạo hàm của hàm sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Dữ liệu cho bài toán AND logic gate
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])  # Đầu vào

y = np.array([[0],  # Output tương ứng với các đầu vào
              [0],
              [0],
              [1]])

# Khởi tạo tham số
np.random.seed(42)

input_layer_size = X.shape[1]  # 2 đặc trưng đầu vào
hidden_layer_size = 4  # Số lượng nơ-ron trong lớp ẩn
output_layer_size = y.shape[1]  # 1 lớp phân loại

# Khởi tạo trọng số ngẫu nhiên
W1 = np.random.rand(input_layer_size, hidden_layer_size)
b1 = np.random.rand(1, hidden_layer_size)

W2 = np.random.rand(hidden_layer_size, output_layer_size)
b2 = np.random.rand(1, output_layer_size)

# Học số (learning rate)
learning_rate = 0.1

# Số vòng lặp (epochs)
epochs = 10000

# Đào tạo MLP
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, W1) + b1  # Tính toán đầu vào lớp ẩn
    hidden_layer_output = sigmoid(hidden_layer_input)  # Tính toán đầu ra lớp ẩn

    output_layer_input = np.dot(hidden_layer_output, W2) + b2  # Tính toán đầu vào lớp output
    predicted_output = sigmoid(output_layer_input)  # Tính toán đầu ra lớp output

    # Tính toán lỗi (loss) - Mean Squared Error (MSE)
    loss = np.mean((predicted_output - y) ** 2)

    # Backpropagation
    # Tính gradient cho lớp output
    d_predicted_output = predicted_output - y
    d_output_layer = d_predicted_output * sigmoid_derivative(predicted_output)  # Đạo hàm của sigmoid

    # Cập nhật trọng số W2 và bias b2
    d_W2 = np.dot(hidden_layer_output.T, d_output_layer)
    d_b2 = np.sum(d_output_layer, axis=0, keepdims=True)
    
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2

    # Tính gradient cho lớp ẩn
    d_hidden_layer = np.dot(d_output_layer, W2.T) * sigmoid_derivative(hidden_layer_output)
    d_W1 = np.dot(X.T, d_hidden_layer)
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)

    # Cập nhật trọng số W1 và bias b1
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

    # In ra loss sau mỗi 1000 vòng lặp
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Đánh giá mô hình
print("\nPredicted Output:")
print(predicted_output)

# In ra kết quả cuối cùng sau huấn luyện
print("\nFinal Output (rounded):")
print(np.round(predicted_output))

