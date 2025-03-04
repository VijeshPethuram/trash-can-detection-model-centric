import numpy as np
import tensorflow as tf

MODEL_INPUT_WIDTH = 48
MODEL_INPUT_HEIGHT = 48
TF_MODEL = "object_recognition"

test_dir = "DATASET/TEST"

# Load the test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    interpolation="bilinear",
    image_size=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
)

def rescale(x, y):
    return (x / 255.0) * 2 - 1, y

test_ds = test_ds.map(rescale)

# Load the TFLite model
with open("model_nano.tflite", "rb") as f:
    tfl_model = f.read()

interpreter = tf.lite.Interpreter(model_content=tfl_model)
interpreter.allocate_tensors()
i_details = interpreter.get_input_details()[0]
o_details = interpreter.get_output_details()[0]

i_quant = i_details["quantization_parameters"]
i_scale = i_quant['scales'][0]
i_zero_point = i_quant['zero_points'][0]

# Evaluate the model
num_correct_samples = 0
num_total_samples = 0

test_ds0 = test_ds.unbatch()

for i_value, o_value in test_ds0.batch(1):
    i_value = (i_value / i_scale) + i_zero_point
    i_value = tf.cast(i_value, dtype=tf.int8)
    interpreter.set_tensor(i_details["index"], i_value)
    interpreter.invoke()
    o_pred = interpreter.get_tensor(o_details["index"])[0]

    if np.argmax(o_pred) == o_value.numpy():
        num_correct_samples += 1
    num_total_samples += 1

print("Accuracy:", num_correct_samples / num_total_samples)
