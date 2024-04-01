import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import flwr as fl

# Set paths to the image directories
train_polyp_dir = r'C:\Users\KUSHAGRA\Documents\Dataset\FigShareCP-CHILD\CP-CHILD-A\Train\Polyp'
train_non_polyp_dir = r'C:\Users\KUSHAGRA\Documents\Dataset\FigShareCP-CHILD\CP-CHILD-A\Train\Non-Polyp'
test_polyp_dir = r'C:\Users\KUSHAGRA\Documents\Dataset\FigShareCP-CHILD\CP-CHILD-A\Test\Polyp'
test_non_polyp_dir = r'C:\Users\KUSHAGRA\Documents\Dataset\FigShareCP-CHILD\CP-CHILD-A\Test\Non-Polyp'

btrain_polyp_dir = r'C:\Users\KUSHAGRA\Documents\Dataset\FigShareCP-CHILD\CP-CHILD-B\Train\Polyp'
btrain_non_polyp_dir = r'C:\Users\KUSHAGRA\Documents\Dataset\FigShareCP-CHILD\CP-CHILD-B\Train\Non-Polyp'
btest_polyp_dir = r'C:\Users\KUSHAGRA\Documents\Dataset\FigShareCP-CHILD\CP-CHILD-B\Test\Polyp'
btest_non_polyp_dir = r'C:\Users\KUSHAGRA\Documents\Dataset\FigShareCP-CHILD\CP-CHILD-B\Test\Non-Polyp'

# Function to load and preprocess images
def load_and_preprocess_images(directory):
    image_list = []
    label_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(directory, filename), target_size=(256, 256 )
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
            image_list.append(img_array)
            if 'Non-Polyp' in directory:
                label_list.append(0)  # Non-Polyp class
            else:
                label_list.append(1)  # Polyp class
    return np.array(image_list), np.array(label_list)

# Load and preprocess training images
X_train_polyp, y_train_polyp = load_and_preprocess_images(train_polyp_dir)
X_train_non_polyp, y_train_non_polyp = load_and_preprocess_images(train_non_polyp_dir)

# Load and preprocess testing images
X_test_polyp, y_test_polyp = load_and_preprocess_images(test_polyp_dir)
X_test_non_polyp, y_test_non_polyp = load_and_preprocess_images(test_non_polyp_dir)

# Concatenate training and testing data
X_train = np.concatenate((X_train_polyp, X_train_non_polyp), axis=0)
y_train = np.concatenate((y_train_polyp, y_train_non_polyp), axis=0)
X_test = np.concatenate((X_test_polyp, X_test_non_polyp), axis=0)
y_test = np.concatenate((y_test_polyp, y_test_non_polyp), axis=0)




# Load and preprocess training images
bX_train_polyp, by_train_polyp = load_and_preprocess_images(btrain_polyp_dir)
bX_train_non_polyp, by_train_non_polyp = load_and_preprocess_images(btrain_non_polyp_dir)

# Load and preprocess testing images
bX_test_polyp, by_test_polyp = load_and_preprocess_images(btest_polyp_dir)
bX_test_non_polyp, by_test_non_polyp = load_and_preprocess_images(btest_non_polyp_dir)

# Concatenate training and testing data
bX_train = np.concatenate((bX_train_polyp, bX_train_non_polyp), axis=0)
by_train = np.concatenate((by_train_polyp, by_train_non_polyp), axis=0)
bX_test = np.concatenate((bX_test_polyp, bX_test_non_polyp), axis=0)
by_test = np.concatenate((by_test_polyp, by_test_non_polyp), axis=0)



X_train = np.concatenate((X_train, bX_train), axis=0)
y_train = np.concatenate((y_train, by_train), axis=0)
X_test = np.concatenate((X_test, bX_test), axis=0)
y_test = np.concatenate((y_test, by_test), axis=0)


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for validation and testing sets (only rescaling)
val_test_datagen = ImageDataGenerator()

# Train data generator
train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=32
)

# Validation data generator
val_generator = val_test_datagen.flow(
    X_val,
    y_val,
    batch_size=32
)

# Test data generator
test_generator = val_test_datagen.flow(
    X_test,
    y_test,
    batch_size=32,
    shuffle=False
)

# Load MobileNet model (pre-trained on ImageNet)
mobilenet_model = tf.keras.applications.MobileNet(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze pre-trained layers
for layer in mobilenet_model.layers:
    layer.trainable = False

# Add classification head
model = tf.keras.models.Sequential([
    mobilenet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(X_val) // 32
)

#Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // 32)
print("Test Accuracy:", test_accuracy)
'''
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        # model.fit(x_train, y_train, epochs=1, batch_size=32)
        model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // 32,
            epochs=5,
            validation_data=val_generator,
            validation_steps=len(X_val) // 32
        )
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # loss, accuracy = model.evaluate(x_test, y_test)
        test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // 32)
        return test_loss, len(X_test), {"accuracy": test_accuracy}


fl.client.start_numpy_client(server_address="127.0.0.1:5000", client=FlowerClient())
'''
