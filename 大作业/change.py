import tensorflow as tf

# 加载已经训练好的模型
model = tf.keras.models.load_model('age_gender_model.h5')

# 将模型转换为 TensorFlow Lite 模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 如果模型包含自定义层，请取消下面的注释
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

# 保存转换后的模型
with open('age_gender_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite 模型已保存为 'age_gender_model.tflite'")
