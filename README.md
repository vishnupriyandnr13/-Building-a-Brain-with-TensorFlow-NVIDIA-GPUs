# 🧠 Building a Brain with TensorFlow & NVIDIA GPUs

> A beginner-friendly deep learning project that demonstrates how an artificial neural network (ANN) can learn to classify images from the Fashion MNIST dataset using **TensorFlow** and **GPU acceleration**.

---

## 📌 Overview

This project explores the fundamentals of **Artificial Neural Networks (ANNs)** by building a simple image classification model using **TensorFlow/Keras**.

The notebook walks through the complete deep learning workflow:

- Understanding image datasets
- Building an artificial neuron
- Creating a neural network
- Training the model
- Evaluating performance
- Understanding how GPUs accelerate deep learning workloads

The project uses the **Fashion MNIST** dataset consisting of grayscale images of clothing items and trains a neural network to recognize different categories automatically.

---

## 🎯 Objectives

- Learn the basics of neural networks
- Understand how image classification works
- Build a neural network using TensorFlow/Keras
- Train and evaluate a machine learning model
- Explore the role of GPUs in accelerating AI workloads

---

## 📂 Dataset

The notebook uses the **Fashion MNIST** dataset provided by TensorFlow.

Dataset contains:

- **60,000 training images**
- **10,000 validation/test images**
- **10 clothing categories**

### Categories

| Label | Class |
|---------|----------------|
| 0 | T-shirt / Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

Each image has dimensions:

```
28 × 28 pixels
```

---

# 🏗️ Model Architecture

The notebook builds a simple neural network using TensorFlow's Sequential API.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10)
])
```

### Components

### 🔹 Flatten Layer

Converts

```
28 × 28 image
```

into

```
784-dimensional vector
```

making it suitable as input to dense neural layers.

---

### 🔹 Dense Layer

A fully connected neural layer that predicts probabilities for the 10 Fashion MNIST classes.

---

## 🚀 Workflow

```
Fashion MNIST Images
            │
            ▼
     Data Preprocessing
            │
            ▼
      Flatten Layer
            │
            ▼
      Dense Neural Layer
            │
            ▼
     Class Predictions
            │
            ▼
      Model Evaluation
```

---

# 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

# ⚡ How NVIDIA Helps in This Project

Deep learning relies heavily on **matrix multiplication** and **parallel computations**.

Instead of executing these operations sequentially on a CPU, **NVIDIA GPUs** perform thousands of computations simultaneously.

## GPU Advantages

- Faster model training
- Efficient matrix operations
- Parallel tensor computations
- Reduced training time
- Better scalability for larger neural networks

In the notebook, GPU availability can be verified using:

```python
import tensorflow as tf

tf.config.list_physical_devices("GPU")
```

If an NVIDIA GPU with CUDA support is available, TensorFlow automatically utilizes it to accelerate training.

---

## Why NVIDIA GPUs Are Ideal for Deep Learning

Neural networks repeatedly perform operations such as:

```
Input Matrix × Weight Matrix
```

These computations involve millions or even billions of arithmetic operations.

NVIDIA GPUs contain thousands of CUDA cores specifically designed for parallel mathematical workloads, making them significantly faster than CPUs for neural network training.

For example:

| Hardware | Relative Training Speed |
|-----------|------------------------|
| CPU | Slow |
| NVIDIA GPU | Significantly Faster |

---

# 📈 Learning Outcomes

By completing this notebook, you will understand:

- Artificial neurons
- Neural networks
- Image classification
- TensorFlow Sequential models
- Fashion MNIST dataset
- GPU acceleration in AI
- Basic deep learning workflows

---

# ▶️ Running the Notebook

## Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
pip install tensorflow matplotlib numpy
```

---

## Launch

```bash
jupyter notebook
```

Open:

```
BuildingABrain.ipynb
```

Run all cells sequentially.

---

# 📊 Sample Output

The model learns to classify images into one of the following categories:

- 👕 T-shirt
- 👖 Trouser
- 🧥 Coat
- 👗 Dress
- 👟 Sneaker
- 👜 Bag
- 👢 Ankle Boot

---

# 📚 References

- TensorFlow Documentation: https://www.tensorflow.org/
- Fashion MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
- NVIDIA Deep Learning Institute: https://www.nvidia.com/en-us/training/

---

# 🙌 Acknowledgements

This notebook is inspired by educational material from the **NVIDIA Deep Learning Institute (DLI)** and demonstrates how GPU-accelerated computing can simplify and speed up the process of building and training neural networks using TensorFlow.

---

## ⭐ If you found this project useful, consider starring the repository and sharing it with others interested in Deep Learning and AI!
