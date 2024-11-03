import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# Load the trained model
model = tf.keras.models.load_model('animal_classifier.keras')

# Function to predict the animal class
def predict_animal(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict the class
    predictions = model.predict(img_array)
    class_indices = ['elephant', 'panda', 'tiger']
    predicted_class = class_indices[np.argmax(predictions[0])]
    return predicted_class
