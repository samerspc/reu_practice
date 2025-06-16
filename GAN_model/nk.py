# Установка необходимых библиотек
!pip install -q tensorflow tensorflow-gpu matplotlib numpy
!pip install -q tensorflow-datasets

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import tensorflow_datasets as tfds

# Параметры модели
IMG_SIZE = 64
BATCH_SIZE = 32
LATENT_DIM = 100
NUM_CLASSES = 101
EPOCHS = 30

# Загрузка и подготовка датасета Food-101
def load_and_prepare_data():
    # Загрузка датасета
    (train_ds, test_ds), ds_info = tfds.load(
        'food101',
        split=['train', 'validation'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    
    # Функция предварительной обработки
    def preprocess(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = (image / 127.5) - 1.0  # Нормализация в диапазон [-1, 1]
        return image, label
    
    # Подготовка данных
    def prepare_dataset(ds):
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000).batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_ds = prepare_dataset(train_ds)
    test_ds = prepare_dataset(test_ds)
    
    return train_ds, test_ds

# Создание генератора (ОКОНЧАТЕЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)
def build_generator():
    noise = layers.Input(shape=(LATENT_DIM,))
    label = layers.Input(shape=(1,), dtype='int32')
    
    # Встраивание метки класса
    label_embedding = layers.Embedding(NUM_CLASSES, LATENT_DIM)(label)
    label_embedding = layers.Flatten()(label_embedding)
    
    # Объединение шума и метки
    x = layers.Concatenate()([noise, label_embedding])
    
    # Увеличиваем размерность перед сверточными слоями
    x = layers.Dense(4*4*1024)(x)
    x = layers.Reshape((4, 4, 1024))(x)
    
    # Последовательность слоев Conv2DTranspose для увеличения размера
    # 4x4 -> 8x8
    x = layers.Conv2DTranspose(512, (4,4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 8x8 -> 16x16
    x = layers.Conv2DTranspose(256, (4,4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 16x16 -> 32x32
    x = layers.Conv2DTranspose(128, (4,4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 32x32 -> 64x64 (добавляем этот слой)
    x = layers.Conv2DTranspose(64, (4,4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Выходной слой
    output = layers.Conv2DTranspose(3, (4,4), padding='same', activation='tanh')(x)
    
    # Проверка размера вывода
    print(f"Generator output shape: {output.shape}")
    
    return models.Model([noise, label], output, name='Generator')

# Создание дискриминатора
def build_discriminator():
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Сверточные блоки
    x = layers.Conv2D(64, (4,4), strides=2, padding='same')(img_input)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(128, (4,4), strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(256, (4,4), strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(512, (4,4), strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Выравнивание и выходы
    x = layers.Flatten()(x)
    
    # Выход для классификации подлинности
    validity = layers.Dense(1, activation='sigmoid')(x)
    
    # Выход для классификации класса
    label = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Проверка размера ввода
    print(f"Discriminator input shape: {img_input.shape}")
    
    return models.Model(img_input, [validity, label], name='Discriminator')

# Функция для генерации и сохранения изображений
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i] + 1) / 2)  # Денормализация
        plt.axis('off')
    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# Загрузка данных
train_ds, test_ds = load_and_prepare_data()

# Создание моделей
generator = build_generator()
discriminator = build_discriminator()

# Проверка размеров
print("Проверка размеров:")
print(f"Generator output shape: {generator.output_shape}")
print(f"Discriminator input shape: {discriminator.input_shape}")

# Оптимизаторы
generator_optimizer = optimizers.Adam(0.0002, 0.5)
discriminator_optimizer = optimizers.Adam(0.0002, 0.5)

# Функции потерь
cross_entropy = tf.keras.losses.BinaryCrossentropy()
sparse_categorical_loss = tf.keras.losses.SparseCategoricalCrossentropy()

# Шаг обучения
@tf.function
def train_step(real_images, real_labels):
    batch_size = tf.shape(real_images)[0]
    
    # Создание шума и меток для генератора
    noise = tf.random.normal([batch_size, LATENT_DIM])
    fake_labels = tf.random.uniform([batch_size, 1], 0, NUM_CLASSES, dtype=tf.int32)
    
    with tf.GradientTape(persistent=True) as tape:
        # Генерация изображений
        generated_images = generator([noise, fake_labels], training=True)
        
        # Предсказания для реальных изображений
        real_validity, real_pred_labels = discriminator(real_images, training=True)
        
        # Предсказания для сгенерированных изображений
        fake_validity, fake_pred_labels = discriminator(generated_images, training=True)
        
        # Потери дискриминатора
        real_loss = cross_entropy(tf.ones_like(real_validity), real_validity)
        fake_loss = cross_entropy(tf.zeros_like(fake_validity), fake_validity)
        disc_loss = real_loss + fake_loss
        
        # Потери классификации
        class_loss = sparse_categorical_loss(real_labels, real_pred_labels)
        
        # Общие потери дискриминатора
        total_disc_loss = disc_loss + class_loss
        
        # Потери генератора
        gen_loss = cross_entropy(tf.ones_like(fake_validity), fake_validity)
        gen_class_loss = sparse_categorical_loss(fake_labels, fake_pred_labels)
        total_gen_loss = gen_loss + gen_class_loss
    
    # Градиенты и обновление весов
    disc_gradients = tape.gradient(total_disc_loss, discriminator.trainable_variables)
    disc_gradients = [tf.clip_by_value(g, -1., 1.) for g in disc_gradients]  # Gradient clipping
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    gen_gradients = tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    
    return total_gen_loss, total_disc_loss

# Запуск обучения
fixed_noise = tf.random.normal([16, LATENT_DIM])
fixed_labels = tf.random.uniform([16, 1], 0, NUM_CLASSES, dtype=tf.int32)

for epoch in range(EPOCHS):
    gen_loss_list = []
    disc_loss_list = []
    
    for real_images, real_labels in tqdm(train_ds):
        gen_loss, disc_loss = train_step(real_images, real_labels)
        gen_loss_list.append(gen_loss)
        disc_loss_list.append(disc_loss)
    
    # Средние потери за эпоху
    avg_gen_loss = tf.reduce_mean(gen_loss_list)
    avg_disc_loss = tf.reduce_mean(disc_loss_list)
    
    print(f'Epoch {epoch+1}/{EPOCHS}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}')
    
    # Генерация и сохранение изображений каждые 5 эпох
    if (epoch + 1) % 5 == 0:
        generate_and_save_images(generator, epoch + 1, [fixed_noise, fixed_labels])

# Сохранение моделей
generator.save('food101_generator.h5')
discriminator.save('food101_discriminator.h5')

# Оценка модели
def evaluate_model(discriminator, test_ds):
    total = 0
    correct = 0
    
    for real_images, real_labels in test_ds:
        # Классификация реальных изображений
        _, real_pred = discriminator(real_images, training=False)
        
        # Точность для реальных изображений
        real_labels = real_labels.numpy()
        real_pred_labels = tf.argmax(real_pred, axis=1).numpy()
        correct += np.sum(real_labels == real_pred_labels)
        total += real_labels.shape[0]
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# Запуск оценки
test_accuracy = evaluate_model(discriminator, test_ds)
print(f'Final Test Accuracy: {test_accuracy:.4f}')

# Визуализация результатов
def plot_results(images, titles):
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow((images[i] + 1) / 2)  # Денормализация
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Примеры реальных изображений
for real_images, _ in train_ds.take(1):
    plot_results(real_images[:5], ['Real Image']*5)

# Примеры сгенерированных изображений
noise = tf.random.normal([5, LATENT_DIM])
labels = tf.random.uniform([5, 1], 0, NUM_CLASSES, dtype=tf.int32)
generated_images = generator([noise, labels], training=False)
plot_results(generated_images, ['Generated']*5)