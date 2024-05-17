from keras.applications.densenet import layers
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
inputs = keras.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, 3)(inputs)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3)(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3)(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x=layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(2, activation='softmax')(x)  # Change the number of units to 2 for binary classification
model1 = keras.Model(inputs=inputs, outputs=outputs)

model1.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

train_data_dir = "D:/Users/henta/Bureau/2 ing/deep learning/tp/FaceMaskDataset/FaceMaskDataset/train224"
validation_data_dir = "D:/Users/henta/Bureau/2 ing/deep learning/tp/FaceMaskDataset/FaceMaskDataset/test224"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Data Generators
batch_size = 128
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode="categorical"
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode="categorical"
)

# Callbacks
checkpoint1 = ModelCheckpoint("facemask2.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=50, verbose=1, mode='auto')

# Model Training
history1 = model1.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[checkpoint1, early]
)

# Serialize weights to HDF5
model1.save("facemask1_final.h5")
# model2.save("facemask2_final.h5")

# Visualize Performances
plt.plot(history1.history['accuracy'], label='train_accuracy')
plt.plot(history1.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

