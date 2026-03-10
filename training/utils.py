from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(data_dir, target_size=(224,224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_gen = train_datagen.flow_from_directory(
        data_dir, target_size=target_size, batch_size=batch_size,
        class_mode='binary', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        data_dir, target_size=target_size, batch_size=batch_size,
        class_mode='binary', subset='validation'
    )
    return train_gen, val_gen