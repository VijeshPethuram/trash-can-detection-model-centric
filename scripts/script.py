# 1. Import Statements
import tensorflow as tf
import json
import os
import cv2
import random
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping

# 2. Data Processing
class TrashDataGenerator:
    def __init__(self, json_path, img_dir, batch_size=8, augment=False):
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.augment = augment

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']
        self.categories = data['categories']
        
        self.category_name_to_index = {cat['supercategory']: idx for idx, cat in enumerate(self.categories)}

        self.num_classes = len(set([cat['supercategory'] for cat in self.categories]))
        self.image_ids = list(self.images.keys())

        self.valid_image_ids = []
        for img_id in self.image_ids:
            img_path = os.path.join(self.img_dir, self.images[img_id]['file_name'])
            if os.path.exists(img_path):
                self.valid_image_ids.append(img_id)

        self.image_ids = self.valid_image_ids

    def __len__(self):
        return max(1, len(self.image_ids) // self.batch_size)

    def augment_image(self, img, boxes):
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        if self.augment:
            if random.random() > 0.5:
                img = tf.image.flip_left_right(img)
                boxes[:, 0] = 1 - boxes[:, 0] - boxes[:, 2]

            if random.random() > 0.5:
                k = random.randint(0, 3) 
                img = tf.image.rot90(img, k=k)
                boxes = self.rotate_bboxes(boxes, k)

            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
            img = tf.image.resize_with_crop_or_pad(img, 224, 224)

        return img, boxes

    def rotate_bboxes(self, boxes, k):
        if k == 1:  
            boxes[:, [0, 1]] = boxes[:, [1, 0]] 
            boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2] 
        elif k == 2: 
            boxes[:, 0] = 1 - boxes[:, 0] - boxes[:, 2]  
            boxes[:, 1] = 1 - boxes[:, 1] - boxes[:, 3]  
        elif k == 3:  
            boxes[:, [0, 1]] = boxes[:, [1, 0]]  
            boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2]  
            boxes[:, 0] = 1 - boxes[:, 0] - boxes[:, 2]  
        return boxes

    def generate_data(self):
        while True:
            np.random.shuffle(self.image_ids)

            for img_id in self.image_ids:
                try:
                    img_info = self.images[img_id]
                    img_path = os.path.join(self.img_dir, img_info['file_name'])

                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))

                    img_anns = [ann for ann in self.annotations if ann['image_id'] == img_id]
                    boxes = []
                    labels = []

                    for ann in img_anns:
                        x, y, w, h = ann['bbox']
                        x = x / img_info['width']
                        y = y / img_info['height']
                        w = w / img_info['width']
                        h = h / img_info['height']

                        boxes.append([x, y, w, h])

                        category_name = self.categories[ann['category_id']]['supercategory']
                        labels.append(self.category_name_to_index[category_name])

                    boxes = np.array(boxes, dtype=np.float32)
                    labels = np.array(labels, dtype=np.int32)

                    labels_one_hot = np.zeros((len(labels), self.num_classes), dtype=np.int32)
                    labels_one_hot[np.arange(len(labels)), labels] = 1

                    if len(boxes) < 28:
                        padding = np.zeros((28 - len(boxes), 4), dtype=np.float32)
                        boxes = np.vstack([boxes, padding])

                    if len(labels_one_hot) < 28:
                        padding = np.zeros((28 - len(labels_one_hot), self.num_classes), dtype=np.int32)
                        labels_one_hot = np.vstack([labels_one_hot, padding])

                    img, boxes = self.augment_image(img, boxes)

                    yield img, (boxes, labels_one_hot)

                except Exception as e:
                    continue

    def create_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(28, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(28, self.num_classes), dtype=tf.int32)
            )
        )

        dataset = tf.data.Dataset.from_generator(
            self.generate_data,
            output_signature=output_signature
        )

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.take(len(self.image_ids) // self.batch_size).cache().repeat()

        return dataset

# 3. Model Architecture
def create_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x_box = Dense(512, activation='relu')(x)
    x_box = Dropout(0.5)(x_box)
    box_output = Dense(28 * 4, activation='sigmoid')(x_box)
    box_output = Reshape((28, 4))(box_output)

    x_class = Dense(512, activation='relu')(x)
    x_class = Dropout(0.5)(x_class)
    class_output = Dense(28 * num_classes, activation='softmax')(x_class)
    class_output = Reshape((28, num_classes))(class_output)

    model = Model(inputs=base_model.input, outputs=[box_output, class_output])

    model.compile(
        optimizer='sgd',
        loss=[bbox_loss, class_loss],
        loss_weights=[1.0, 1.0]
    )

    return model

def bbox_loss(y_true, y_pred):
    loss = tf.abs(y_true - y_pred)
    loss = tf.where(loss < 1.0, 0.5 * tf.square(loss), loss - 0.5)
    return tf.reduce_mean(loss)

def class_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

# 4. Model Training
def train_model():
    try:
        annotations_path = '../data/annotations.json'  
        img_dir = '../data'

        with open(annotations_path, 'r') as f:
            data = json.load(f)

        num_images = len(data['images'])
        train_size = int(0.8 * num_images)
        valid_size = int(0.1 * num_images)
        test_size = num_images - train_size - valid_size

        train_images = data['images'][:train_size]
        valid_images = data['images'][train_size:train_size + valid_size]
        test_images = data['images'][train_size + valid_size:]

        train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in [img['id'] for img in train_images]]
        valid_annotations = [ann for ann in data['annotations'] if ann['image_id'] in [img['id'] for img in valid_images]]
        test_annotations = [ann for ann in data['annotations'] if ann['image_id'] in [img['id'] for img in test_images]]

        os.makedirs('../data/split', exist_ok=True) 

        for split, annotations in zip(['train', 'valid', 'test'], [train_annotations, valid_annotations, test_annotations]):
            split_data = {
                'images': [img for img in data['images'] if img['id'] in [ann['image_id'] for ann in annotations]],
                'annotations': annotations,
                'categories': data['categories']
            }
            with open(f'../data/split/{split}_annotations.json', 'w') as f:
                json.dump(split_data, f)

        train_generator = TrashDataGenerator(
            '../data/split/train_annotations.json',
            img_dir,
            batch_size=8,
            augment=True
        )
        valid_generator = TrashDataGenerator(
            '../data/split/valid_annotations.json',
            img_dir,
            batch_size=8,
            augment=False
        )

        train_dataset = train_generator.create_dataset()
        val_dataset = valid_generator.create_dataset()

        train_steps_per_epoch = len(train_generator.image_ids) // train_generator.batch_size
        valid_steps_per_epoch = len(valid_generator.image_ids) // valid_generator.batch_size

        num_classes = train_generator.num_classes

        model = create_model(num_classes)
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=valid_steps_per_epoch,
            callbacks=[early_stopping]
        )

        save_model(model)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

# 5. Model Generation
def save_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    try:
        tflite_model = converter.convert()
        os.makedirs('../models', exist_ok=True)
        with open('../models/trash_detection_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("Model successfully converted and saved to TFLite format")
        return True
    except Exception as e:
        print(f"Error during TFLite conversion: {str(e)}")
        raise

# 6. Main Execution
if __name__ == '__main__':
    train_model()