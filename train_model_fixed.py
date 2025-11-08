import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def check_gpu_availability():
    """Check if GPU is available and configure TensorFlow accordingly"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} devices")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPU available, using CPU")
        return False

def create_simpler_model():
    """Create a simpler model that's less prone to compatibility issues"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    return model

def create_very_simple_model():
    """Create a very simple model as fallback"""
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

def load_and_preprocess_data():
    """Load and preprocess MNIST data"""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"Data loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples")
    return (x_train, y_train), (x_test, y_test)

def train_model():
    """Main training function"""
    print("Starting MNIST model training...")
    
    # Check GPU availability
    check_gpu_availability()
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Try simpler model first
    try:
        print("Creating CNN model...")
        model = create_simpler_model()
    except Exception as e:
        print(f"CNN model failed, using simple DNN: {e}")
        model = create_very_simple_model()
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Train with smaller batch size and fewer epochs for stability
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        batch_size=64,  # Smaller batch size for stability
        epochs=5,       # Fewer epochs for quick testing
        validation_data=(x_test, y_test),
        verbose=1,
        shuffle=True
    )
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save the model
    model.save('mnist_cnn_model.h5')
    print("Model saved as 'mnist_cnn_model.h5'")
    
    # Plot training history
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        print("Training history plot saved")
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    return model

if __name__ == "__main__":
    try:
        model = train_model()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        print("\n Troubleshooting tips:")
        print("1. Try: pip install tensorflow-macos tensorflow-metal")
        print("2. Try using conda: conda install tensorflow")
        print("3. Reduce batch size in the code")
        print("4. Try CPU-only: pip install tensorflow-cpu")