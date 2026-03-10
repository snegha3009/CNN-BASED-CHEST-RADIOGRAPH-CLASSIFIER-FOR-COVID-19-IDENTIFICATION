from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils import create_generators

train_gen, val_gen = create_generators('../chest_xray_dataset', target_size=(224,224))

base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base.layers: layer.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [ModelCheckpoint(r'../backend/models/resnet50_covid.h5', save_best_only=True),
             EarlyStopping(patience=5, restore_best_weights=True),
             ReduceLROnPlateau(patience=3)]

model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks)