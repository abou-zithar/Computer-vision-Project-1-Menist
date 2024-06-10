# MNIST CNN Models

This repository contains two Convolutional Neural Network (CNN) models implemented in TensorFlow and Keras to classify the MNIST dataset. The first model uses a functional approach, and the second model uses a class-based approach by inheriting from `tf.keras.Model`.

## Models

### Functional Model

The functional model is defined using the Keras functional API, which is more flexible and allows for the creation and reuse of parts of the model.

#### Code

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

def functional_model():
    input = Input(shape=(28,28,1))
     
    x = Conv2D(32, (3,3), activation='relu')(input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    return model
```
### Custom Model
The custom model is defined by creating a class that inherits from tf.keras.Model, providing more control and customization.
```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

class MyCustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
```

### Training and Evaluation
The models are trained and evaluated on the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits.

#### Data Loading and Preparation

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, Y_train), (x_test, Y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape the data to include channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encode the labels
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
```
### Training

```python

# Instantiate and compile the model
model = MyCustomModel()  # or functional_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, Y_train, batch_size=64, epochs=3, validation_split=0.2)

```
### Evaluation
```python
# Evaluate the model on the test set
model.evaluate(x_test, Y_test, batch_size=64)

# Generate predictions
y_pred = model.predict(x_test, batch_size=64)
```

### Utilities
A utility function to display some examples from the dataset is provided in My_utils.

#### Display Examples
```python
from My_utils import display_some_example

# Display some examples from the dataset
display_some_example(x_train, Y_train)
```
### License
This project is licensed under the MIT License.

### Acknowledgements
- The MNIST dataset is provided by Yann LeCun and Corinna Cortes.
- TensorFlow and Keras libraries are developed and maintained by the TensorFlow team.
