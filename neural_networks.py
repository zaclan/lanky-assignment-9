import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

# Create a directory to store results
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Activation functions and their derivatives
def tanh(z):
    """Hyperbolic tangent activation function."""
    return np.tanh(z)

def tanh_derivative(z):
    """Derivative of the tanh activation function."""
    return 1 - np.tanh(z) ** 2

def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of the ReLU activation function."""
    return (z > 0).astype(float)

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the sigmoid activation function."""
    s = sigmoid(z)
    return s * (1 - s)

# Activation ranges for plotting purposes
activation_ranges = {
    'tanh': (-1, 1),
    'sigmoid': (0, 1),
    'relu': (0, None)  # None indicates no upper bound
}

# Define the MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, activation='tanh'):
        np.random.seed(0)
        self.learning_rate = learning_rate
        self.activation_name = activation

        # Define activation functions and their derivatives
        if self.activation_name == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif self.activation_name == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif self.activation_name == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError("Unsupported activation function.")

        # Initialize weights and biases with appropriate strategies
        if self.activation_name == 'relu':
            # He initialization for ReLU activation
            limit1 = np.sqrt(2 / input_dim)
            self.W1 = np.random.normal(0, limit1, size=(input_dim, hidden_dim))
            self.b1 = np.zeros((1, hidden_dim))

            limit2 = np.sqrt(2 / hidden_dim)
            self.W2 = np.random.normal(0, limit2, size=(hidden_dim, output_dim))
            self.b2 = np.zeros((1, output_dim))
        else:
            # Xavier initialization for tanh and sigmoid activation
            limit1 = np.sqrt(6 / (input_dim + hidden_dim))
            self.W1 = np.random.uniform(-limit1, limit1, size=(input_dim, hidden_dim))
            self.b1 = np.zeros((1, hidden_dim))

            limit2 = np.sqrt(6 / (hidden_dim + output_dim))
            self.W2 = np.random.uniform(-limit2, limit2, size=(hidden_dim, output_dim))
            self.b2 = np.zeros((1, output_dim))

        # For storing activations and gradients for visualization
        self.hidden_layer_input = None
        self.hidden_layer_output = None
        self.output_layer_input = None
        self.output_layer_output = None

        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None

    def forward(self, X):
        # Forward pass
        self.hidden_layer_input = np.dot(X, self.W1) + self.b1  # (n_samples, hidden_dim)
        self.hidden_layer_output = self.activation(self.hidden_layer_input)  # (n_samples, hidden_dim)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.W2) + self.b2  # (n_samples, output_dim)
        self.output_layer_output = self.output_layer_input  # (n_samples, output_dim)
        # Store activations for visualization (if needed)
        out = self.output_layer_output
        return out


    def backward(self, X, y):
        m = X.shape[0]  # number of samples
        # Compute gradients using chain rule
        d_output_layer_input = self.output_layer_output - y  
        self.grad_W2 = np.dot(self.hidden_layer_output.T, d_output_layer_input) / m  
        self.grad_b2 = np.sum(d_output_layer_input, axis=0, keepdims=True) / m  
        d_hidden_layer_output = np.dot(d_output_layer_input, self.W2.T)  
        d_hidden_layer_input = d_hidden_layer_output * self.activation_derivative(self.hidden_layer_input)  
        self.grad_W1 = np.dot(X.T, d_hidden_layer_input) / m 
        self.grad_b1 = np.sum(d_hidden_layer_input, axis=0, keepdims=True) / m  
        
        # Update weights with gradient descent
        self.W1 -= self.learning_rate * self.grad_W1
        self.b1 -= self.learning_rate * self.grad_b1
        self.W2 -= self.learning_rate * self.grad_W2
        self.b2 -= self.learning_rate * self.grad_b2
        # Gradients are stored for visualization (if needed)


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y


# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform several training steps
    for _ in range(10):
        logits = mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden layer activations
    hidden_features = mlp.hidden_layer_output

    # Set activation ranges for plotting
    activation_min, activation_max = activation_ranges[mlp.activation_name]
    if activation_min is None:
        activation_min = hidden_features.min() - 0.1
    if activation_max is None:
        activation_max = hidden_features.max() + 0.1

    # Plot transformed grid lines in the hidden layer feature space
    # 1. Create a grid in the input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    grid_size = 20  # Adjust grid size as needed for resolution
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))
    grid_input = np.c_[xx.ravel(), yy.ravel()]  # Shape: (grid_size*grid_size, 2)

    # 2. Pass the grid through the network up to the hidden layer
    Z1_grid = np.dot(grid_input, mlp.W1) + mlp.b1  # Pre-activation of hidden layer
    A1_grid = mlp.activation(Z1_grid)  # Activation of hidden layer

    # 3. Reshape the transformed grid for plotting
    A1_grid_reshaped = A1_grid.reshape(grid_size, grid_size, -1)  # Shape: (grid_size, grid_size, hidden_dim)

    # 4. Plot the transformed grid lines in the hidden space
    for i in range(grid_size):
        # Plot lines along the x-direction
        ax_hidden.plot(A1_grid_reshaped[i, :, 0], A1_grid_reshaped[i, :, 1], A1_grid_reshaped[i, :, 2],
                       color='lightgray', alpha=0.5)
        # Plot lines along the y-direction
        ax_hidden.plot(A1_grid_reshaped[:, i, 0], A1_grid_reshaped[:, i, 1], A1_grid_reshaped[:, i, 2],
                       color='lightgray', alpha=0.5)

    # Scatter plot of hidden features
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)

    # Plot decision hyperplane in hidden space
    W2 = mlp.W2.squeeze()
    b2 = mlp.b2.squeeze()

    # Create grid in hidden space for plotting the hyperplane
    h1 = np.linspace(activation_min, activation_max, 10)
    h2 = np.linspace(activation_min, activation_max, 10)
    H1, H2 = np.meshgrid(h1, h2)

    # Solve for H3 in the hyperplane equation
    if W2[2] != 0:
        H3 = (-W2[0] * H1 - W2[1] * H2 - b2) / W2[2]
        ax_hidden.plot_surface(H1, H2, H3, alpha=0.3, color='yellow')
    else:
        # Handle cases where W2[2] is zero
        pass  # For simplicity, skip plotting in this case

    ax_hidden.set_title(f"Hidden Layer Feature Space\nStep {frame * 10}")
    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')

    ax_hidden.set_xlim(activation_min, activation_max)
    ax_hidden.set_ylim(activation_min, activation_max)
    ax_hidden.set_zlim(activation_min, activation_max)

    # Plot decision boundary in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx_input, yy_input = np.meshgrid(np.linspace(x_min, x_max, 200),
                                     np.linspace(y_min, y_max, 200))
    grid_input = np.c_[xx_input.ravel(), yy_input.ravel()]
    logits = mlp.forward(grid_input)
    probs = sigmoid(logits).reshape(xx_input.shape)
    ax_input.contourf(xx_input, yy_input, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    ax_input.contour(xx_input, yy_input, probs, levels=[0.5], colors='black')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Decision Boundary in Input Space\nStep {frame * 10}")
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')

    # Visualize network gradients
    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-0.5, 1.5)
    ax_gradient.axis('off')

    # Positions of neurons
    input_neurons = [(0, 0.75), (0, 0.25)]  # x1 at y=0.75, x2 at y=0.25
    hidden_neurons = [(1, 1.0), (1, 0.5), (1, 0.0)]  # h1 at y=1.0, h2 at y=0.5, h3 at y=0.0
    output_neuron = (2, 0.5)

    # Draw neurons and add labels
    # Input neurons
    input_labels = ['x1', 'x2']
    for idx, pos in enumerate(input_neurons):
        circle = Circle(pos, 0.05, color='lightblue', ec='k', zorder=4)
        ax_gradient.add_artist(circle)
        # Add labels
        ax_gradient.text(pos[0] - 0.1, pos[1], input_labels[idx], fontsize=12, ha='right', va='center')

    # Hidden neurons
    hidden_labels = ['h1', 'h2', 'h3']
    for idx, pos in enumerate(hidden_neurons):
        circle = Circle(pos, 0.05, color='lightgreen', ec='k', zorder=4)
        ax_gradient.add_artist(circle)
        # Add labels
        ax_gradient.text(pos[0], pos[1] + 0.1, hidden_labels[idx], fontsize=12, ha='center', va='bottom')

    # Output neuron
    circle = Circle(output_neuron, 0.05, color='salmon', ec='k', zorder=4)
    ax_gradient.add_artist(circle)
    # Add label
    ax_gradient.text(output_neuron[0] + 0.1, output_neuron[1], 'y', fontsize=12, ha='left', va='center')

    # Draw edges with thickness proportional to gradient magnitudes
    # From input to hidden
    max_grad_W1 = np.max(np.abs(mlp.grad_W1))
    for i, input_pos in enumerate(input_neurons):
        for j, hidden_pos in enumerate(hidden_neurons):
            grad = abs(mlp.grad_W1[i, j])
            linewidth = (grad / (max_grad_W1 + 1e-8)) * 5  # Scale for visualization
            ax_gradient.plot([input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]],
                             'k-', linewidth=linewidth)

    # From hidden to output
    max_grad_W2 = np.max(np.abs(mlp.grad_W2))
    for j, hidden_pos in enumerate(hidden_neurons):
        grad = abs(mlp.grad_W2[j, 0])
        linewidth = (grad / (max_grad_W2 + 1e-8)) * 5
        ax_gradient.plot([hidden_pos[0], output_neuron[0]], [hidden_pos[1], output_neuron[1]],
                         'k-', linewidth=linewidth)

    ax_gradient.set_title(f"Gradient Magnitudes\nStep {frame * 10}")

def visualize(activation, learning_rate, step_num):
    """
    Visualizes the training process of the neural network.
    """
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, learning_rate=learning_rate, activation=activation)

    # Set up visualization
    matplotlib.use('agg')  # Use non-interactive backend
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "sigmoid2.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "sigmoid"  # Choose from 'tanh', 'relu', 'sigmoid'
    learning_rate = 0.3
    step_num = 1500
    visualize(activation, learning_rate, step_num)
