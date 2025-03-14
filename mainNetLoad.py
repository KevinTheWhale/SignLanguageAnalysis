# Load CNN and check precision of new data I made myself
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import cv2 # to convert my own images into data


# load the image
image = cv2.imread('IMG_7327.png',cv2.IMREAD_GRAYSCALE)

# resize it to 28x28 scale
image = cv2.resize(image,(28,28))

# Normalize
image = image / 255.0

# Flatten it to be a row entry
image_flattened = image.flatten()

# create df
img_df = pd.DataFrame([image_flattened])

# Queue to csv
img_df.to_csv("my_hand.csv",index=False,header=False)

# load the csv
img_load = pd.read_csv("my_hand.csv", header=None)

# Convert to numpy array
X_test_new = img_load.values.reshape(1,28,28,1)
# Load the model
model = load_model('ASL_sample.keras')

prediction = model.predict(X_test_new)
pred_label = np.argmax(prediction)

print(pred_label) # e

'''
Conclusion:
Predicted incorrectly likely due to real-world factors such as:
    1. Hand angle orientation
    2. Skin tone?
    3. Shadows or different backgrounds
    4. Small dataset (Only had 27455 - Seen as moderate, but all of them had uniformly the same background conditions and lightning)
'''