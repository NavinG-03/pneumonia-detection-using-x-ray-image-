from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: normal, viral, bacterial
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


train_dir = 'dataset/train'
val_dir = 'dataset/val'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=16, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=(150,150), batch_size=16, class_mode='categorical')

# Train model
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save model
model.save('model/pneumonia_cnn_model.h5')
