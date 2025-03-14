# Convolution Neural Network
import numpy
import matplotlib.pyplot as mp
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split


# Load the CSV files
X_train = pd.read_csv('sign_mnist_train.csv')
X_test = pd.read_csv('sign_mnist_test.csv')



################################### Data pre-processing ###################################

# Separate features and labels (assuming the first column is the label)(label = target output)
y_train = X_train.pop('label')  # Extract labels from the training set
y_test = X_test.pop('label')    # Extract labels from the test set

# Convert to NumPy arrays and reshape the data
X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)  # Convert to NumPy and reshape
X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)    # Same for test set


# Normalize pixel values to range 0 to 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=26)  # Assuming 25 classes
y_test = to_categorical(y_test, num_classes=26)


###########################################################################################
################################## Establish Neural Network ###############################

'''
Network Components:

1. Conv2D -- Convolutional Layer (filters,(a,b),activation,input_shape)
            - filters -- # of filters (kernels) that scan the image based on 2^n bit-rate, each filter learns a different pattern / looks for different features,
                         each filter will perform a convolution operation (sliding window operation) with its own set of learned values (weights) which are initialized
                         randomly at first, but during training, they are ADJUSTED based on the LOSS FUNCTION using backpropagation (activation functions with ReLu)
                         Each filter is independent of each other and is updated based on its own output during the convolution.
                         ex) filter 1 might end up detecting edges while filter 2 detect textures or colors of an image.
                         For filters means capturing more details, but at the cost of computation speed and risk of overfitting.

            - (a,b)(size of each filter/kernel) -- Smaller for finer details or smaller images but too small will probably not catch any details and will not learn.
                                                   Larger to cover more area, but at the risk of overfitting or not learning if too large.
            
            - Activation function type -- dictates how it traverses in the network between layers. Typically we just use 'ReLu'

            - input_shape = (l,w,s) -- dimensions of the image (l,w) with type of color scale (s)
                                       1 - Grayscale 
                                       3 - RGB (Standard color gradients R - 0:255, G - 0:255, B - 0:255)
                                       4 - RGBA  (Color gradients now with alpha for light/dark contrasting)

2. MaxPooling2D -- Downsampling operation in CNNs. Reduces the spatial size of feature maps, making the model more efficient and robust to SMALL variations in the input
                    - Reduces Computation -- Fewer pixels means faster processing
                    - Prevents Overfitting -- Removes unnecessary details
                    - Extracts dominant features -- Keeps the most important parts of an image
                    - (2x2) = Widely used

3. Flatten -- Layer that converts 2D feature maps from the convolutional layers into 1D vector, so that it can be fed into the fully connected (DENSE) layers.
              Dense layers require a 1D vector to be READ.

4. Dense

5. Dropout

'''
# Define CNN -- 2 Layer model
model = Sequential([
    # First Conv Layer
    Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)), # Goes in increments of 2^n, n - natural number including 0.
    MaxPooling2D(2,2),

    # Second Conv Layer
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # Flatten and Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(26, activation='softmax')  # 26 output classes
])


model.compile(optimizer = 'adam',
              loss='categorical_crossentropy',
              metrics = ['accuracy'])



###########################################################################################
################################## Train the Model ########################################
'''
Model Components:

1. epochs --

2. batch_size --

3. validation_data --

4. random_state --

'''
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train,
                    epochs = 20,
                    batch_size=32,
                    validation_data = (X_val, y_val))


###########################################################################################
################################## Test the Model #########################################
'''
Model Components:

'''
test_loss, test_accuracy = model.evaluate(X_test, y_test)

###########################################################################################
################################## Save the Model #########################################

model.save("ASL_Sample.keras")

###########################################################################################
################################## Display Results ########################################

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
