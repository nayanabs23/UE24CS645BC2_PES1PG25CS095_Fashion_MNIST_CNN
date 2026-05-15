from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Training Images Shape:", x_train.shape)
print("Training Labels Shape:", y_train.shape)

# Show first image
plt.imshow(x_train[0], cmap='gray')

plt.title(f"Label: {y_train[0]}")

plt.show()