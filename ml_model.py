print("Model Script")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import os

# Load VGG16 without top layers
def create_fine_tuned_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the convolutional base
    base_model.trainable = False

    # Add custom top layers
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: panda, elephant, tiger
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    'animal_classifier_checkpoint.keras',  # Save the model as 'animal_classifier_checkpoint.h5'
    save_best_only=True,  # Save only the best model (based on validation accuracy)
    monitor='val_accuracy',  # Monitor the validation accuracy
    mode='max',  # We want the maximum validation accuracy
    verbose=1  # Print a message when saving the model
)

# Train the model
def train_model():
    model = create_fine_tuned_model()

    # Data augmentation for the training set
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Load the dataset
    train_generator = train_datagen.flow_from_directory(
        r'C:\Users\Adox\animal_species\dataset\train',  # Path to your train dataset folder
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = train_datagen.flow_from_directory(
        r'C:\Users\Adox\animal_species\dataset\test',  # Path to your test dataset folder
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Train the model with the checkpoint callback
    model.fit(
        train_generator, 
        epochs=3, 
        validation_data=validation_generator, 
        callbacks=[checkpoint_callback]
        )

    # Save the fine-tuned model
    model.save('animal_classifier.keras')
    print("Model saved as animal_classifier.keras")
    print("Current working directory:", os.getcwd())
    print("Training model...") 

def evaluate_model():
    # Load the trained model
    model = tf.keras.models.load_model('animal_classifier.keras')

    # Path to your test dataset folder
    test_data_dir = r'C:\\Users\\Adox\\animal_species\\dataset\\test'
    
    # ImageDataGenerator for the test dataset (rescale images, no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow the test data from the directory
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),  # same size as your training images
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # no shuffle so we get results in order
    )
    
    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    # Print test results
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Loss: {test_loss}')

# Function to predict using the fine-tuned model
def predict_animal(img_path):
    model = tf.keras.models.load_model('animal_classifier.keras')

    # Preprocess the input image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalize image

    # Make predictions
    predictions = model.predict(img_array)
    class_indices = ['panda', 'elephant', 'tiger']
    predicted_class = class_indices[tf.argmax(predictions[0])]
    return predicted_class

if __name__ == "__main__":
    train_model()  # Train and save the model
    evaluate_model()  # Evaluate the model after training
