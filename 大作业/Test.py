import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# 数据集路径
dataset_path = r"D:\Desktop\Test\dataset"

# 数据预处理函数
def load_data():
    data = []
    labels = []
    for age_folder in tqdm(os.listdir(dataset_path), desc="加载数据中"):
        age_path = os.path.join(dataset_path, age_folder)
        for gender_folder in os.listdir(age_path):
            gender_path = os.path.join(age_path, gender_folder)
            for image_file in tqdm(os.listdir(gender_path), desc=f"处理 {age_folder}-{gender_folder}", leave=False):
                image_path = os.path.join(gender_path, image_file)
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
                image = tf.keras.preprocessing.image.img_to_array(image)
                data.append(image)
                labels.append([int(age_folder), int(gender_folder) - 111])
    return np.array(data), np.array(labels)

# 加载数据
data, labels = load_data()

# 检查数据和标签的长度是否一致
assert len(data) == len(labels), "数据和标签的长度不一致"

# 划分训练集和测试集
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# 数据归一化
trainX = trainX / 255.0
testX = testX / 255.0

# 标签编码
lb_age = LabelBinarizer()
trainY_age = lb_age.fit_transform(trainY[:, 0])
testY_age = lb_age.transform(testY[:, 0])

trainY_gender = to_categorical(trainY[:, 1])
testY_gender = to_categorical(testY[:, 1])

# 检查训练集和测试集的长度是否一致
assert trainX.shape[0] == trainY_age.shape[0] == trainY_gender.shape[0], "训练数据和标签的长度不一致"
assert testX.shape[0] == testY_age.shape[0] == testY_gender.shape[0], "测试数据和标签的长度不一致"

# 数据增强
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# 构建模型
def build_model():
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(lb_age.classes_), activation='softmax', name='age_output'),
        Dense(2, activation='softmax', name='gender_output')
    ])
    return model

# 编译模型
model = build_model()
model.compile(optimizer='adam', loss={'age_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'}, metrics=['accuracy'])

# 训练模型，带进度条
epochs = 50
for epoch in tqdm(range(epochs), desc="训练模型中"):
    model.fit(
        aug.flow(trainX, {'age_output': trainY_age, 'gender_output': trainY_gender}, batch_size=32),
        validation_data=(testX, {'age_output': testY_age, 'gender_output': testY_gender}),
        epochs=1, verbose=1
    )

# 模型评估，带进度条
results = model.evaluate(testX, {'age_output': testY_age, 'gender_output': testY_gender}, verbose=1)
print(f"Age Loss: {results[1]}, Age Accuracy: {results[3]}")
print(f"Gender Loss: {results[2]}, Gender Accuracy: {results[4]}")

# 保存模型
model.save("age_gender_model.h5")
