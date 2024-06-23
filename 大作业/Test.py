import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 数据集路径
data_dir = 'data'  # 修改为你的data文件夹路径


# 从文件名提取标签
def extract_labels(filename):
    parts = filename.split('-')
    age = int(parts[0])
    gender = 1 if parts[1] == '112' else 0  # 假设 '112' 为 female，其他为 male
    return age, gender


# 创建数据列表
data = []
labels = []

for filename in os.listdir(data_dir):
    if filename.endswith(".jpg"):
        age, gender = extract_labels(filename)
        data.append(os.path.join(data_dir, filename))
        labels.append((age, gender))

# 转换为DataFrame以便于处理
df = pd.DataFrame({'filename': data, 'age': [x[0] for x in labels], 'gender': [x[1] for x in labels]})

# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 数据生成器
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=20
)

val_datagen = ImageDataGenerator(rescale=1. / 255)


# 自定义数据生成器
def create_generator(df, datagen, batch_size=32, target_size=(224, 224)):
    def generator():
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_df = df[start:end]
            images = []
            ages = []
            genders = []
            for _, row in batch_df.iterrows():
                img = tf.keras.preprocessing.image.load_img(row['filename'], target_size=target_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = datagen.random_transform(img_array)
                img_array = datagen.standardize(img_array)
                images.append(img_array)
                ages.append(row['age'])
                genders.append(row['gender'])
            yield np.array(images), {"gender": np.array(genders), "age": np.array(ages)}

    return generator


train_generator = create_generator(train_df, train_datagen)
val_generator = create_generator(val_df, val_datagen)

# 使用tf.data.Dataset从生成器中创建数据集
train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        {"gender": tf.TensorSpec(shape=(None,), dtype=tf.float32),
         "age": tf.TensorSpec(shape=(None,), dtype=tf.float32)}
    )
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    val_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        {"gender": tf.TensorSpec(shape=(None,), dtype=tf.float32),
         "age": tf.TensorSpec(shape=(None,), dtype=tf.float32)}
    )
).prefetch(tf.data.AUTOTUNE)

# 构建模型，指定本地权重文件路径
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'  # 修改为你的本地权重文件路径
base_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights=weights_path)
base_model.trainable = False  # 冻结预训练模型的卷积层

# 添加自定义层
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# 性别输出
gender_output = layers.Dense(1, activation='sigmoid', name='gender')(x)

# 年龄输出
age_output = layers.Dense(1, name='age')(x)

model = models.Model(inputs, [gender_output, age_output])

# 编译模型
model.compile(
    optimizer=optimizers.Adam(),
    loss={
        'gender': 'binary_crossentropy',
        'age': 'mean_squared_error'
    },
    metrics={
        'gender': 'accuracy',
        'age': 'mae'
    }
)

# 训练模型
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100ll,  # 测试时可以先用较少的epochs
    steps_per_epoch=100,  # 可以先用较少的steps_per_epoch进行测试
    validation_steps=50  # 可以先用较少的validation_steps进行测试
)

# 评估模型
results = model.evaluate(val_dataset, steps=50)
print("Evaluation results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value}")


# 使用 tf.function 包装模型，并明确设置输入签名
@tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
def serve_model(inputs):
    return model(inputs)


# 将模型转换为 TensorFlow Lite 模型
converter = tf.lite.TFLiteConverter.from_concrete_functions([serve_model.get_concrete_function()])

tflite_model = converter.convert()

# 保存转换后的模型
with open('age_gender_model_1.tflite', 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite 模型已保存为 'age_gender_model_1.tflite'")
