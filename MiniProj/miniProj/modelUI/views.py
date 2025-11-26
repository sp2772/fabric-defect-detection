from django.shortcuts import render
import numpy as np
import cv2
import time
import tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.staticfiles import finders
import os
from django.conf import settings

# Load only one model globally
model_path = "/mnt/c/modelFiles/10splitsFlatten/quantized_kfold_model_0.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]['quantization']

fabricClass = ['No_Defect','Horizontal Line','Stain','Tear','Vertical Line']
dictClass = ['No_Defect','horizontal_line','stain','tear','vertical_line']
thresholds_path = os.path.join(settings.BASE_DIR, 'modelUI', 'static', 'optimal_thresholds.npy')
thresholds = np.load(thresholds_path, allow_pickle=True).item()
thresholdsDict = {k: float(v) for k, v in thresholds.items()}

@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({"error": "No image uploaded"}, status=400)

        try:
            # Load and preprocess image
            file_bytes = image_file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            stacked = np.stack([img], axis=-1)

            # Quantize image
            img_input = stacked / input_scale + input_zero_point
            img_input = np.clip(img_input, -128, 127).astype(np.int8)
            img_input = np.expand_dims(img_input, axis=0)

            # Inference
            t1 = time.time()
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: [num_classes]
            t2 = time.time()

            class_index = int(np.argmax(output_data))
            class_score = output_data[class_index]
            threshold = thresholds[dictClass[class_index]]

            if class_score >= threshold:
                predicted_class = fabricClass[class_index]
            else:
                predicted_class = "Uncertain"

            inference_time = round(t2 - t1, 4)

            return JsonResponse({
                "prediction": predicted_class,
                "inference_time": inference_time
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST method allowed."}, status=405)

def home(request):
    return render(request, 'home.html')

@csrf_exempt
def get_thresholds(request):
    return JsonResponse(thresholdsDict)
