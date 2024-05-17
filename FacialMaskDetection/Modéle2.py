from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping


train_data_dir ="D:/Users/henta/Bureau/2 ing/deep learning/tp/FaceMaskDataset/FaceMaskDataset/train224"
validation_data_dir = "D:/Users/henta/Bureau/2 ing/deep learning/tp/FaceMaskDataset/FaceMaskDataset/test224"

# Définition du modèle 2
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(2, activation='softmax'))
# Compiler le modèle
model2.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Prétraitement des images
# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255)
# Charger les données d'entraînement et de test

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (64, 64),
batch_size = 128,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (64, 64),
batch_size=128,
class_mode = "categorical")

#affichage des indices des classes
print(train_generator.class_indices)
print(train_generator.samples)       # Print total samples per class

# Entraîner le modèle

# Save the model according to the conditions
checkpoint1 = ModelCheckpoint("facemask2.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=50, verbose=1, mode='auto')

# Train the model

history2 = model2.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks = [checkpoint1, early])

# serialize weights to HDF5
print("Saved model to disk")


# Visualiser les performances
plt.plot(history2.history['accuracy'], label='train_accuracy')
plt.plot(history2.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


