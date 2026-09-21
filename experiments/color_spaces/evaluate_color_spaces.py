import cv2
import os
import random
import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

def load_frames(input_dir, target_size=(224, 224)):
    frames = []
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith(".npy"):
            frame = np.load(os.path.join(input_dir, file_name))
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
    return frames
    
input_dir = "/media/beegfs/home/v582/v582962/Pablo/YUVFeedTest/YUVFrames/"

input_dirs = [subdir for subdir in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, subdir))]

frames_loaded = []
for input_dir_idx in range(len(input_dirs)):
    frames = load_frames(input_dir+input_dirs[input_dir_idx])
    random_frame_index = random.randint(0, len(frames) - 1)
    frames_loaded.append(frames[random_frame_index])

print(frames_loaded[0].shape)

frame_rgb = [cv2.cvtColor(frame, cv2.COLOR_YUV2BGR) for frame in frames_loaded]
# Preprocess the RGB image for MobileNet
preprocessed_images = [preprocess_input(frame) for frame in frames_loaded]
preprocessed_rgb = [preprocess_input(frame) for frame in frame_rgb]

# Load the MobileNet model
model = MobileNet(weights='imagenet')

"""for preprocessed_image in preprocessed_images:
    print('------------------------')
    predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
    
    # Imprimir las 5 clases predichas principales
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    for class_id, class_name, class_prob in decoded_predictions:
        print(f'{class_name}: {class_prob}')
        
for preprocessed_image in preprocessed_rgb:
    print('------------------------')
    predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
    
    # Imprimir las 5 clases predichas principales
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    for class_id, class_name, class_prob in decoded_predictions:
        print(f'{class_name}: {class_prob}')"""
        
import matplotlib.pyplot as plt

def reescalar_matriz(matriz):
    # Encuentra el valor máximo y mínimo en la matriz
    max_valor = np.max(matriz)
    min_valor = np.min(matriz)

    # Calcula la diferencia entre el máximo y el mínimo
    rango = max_valor - min_valor

    # Reescala los valores de la matriz al rango [0, 1]
    matriz_reescalada = (matriz - min_valor) / rango

    return matriz_reescalada

# Crear una función para plotear y guardar las imágenes con las clases predichas
def plot_and_save_images(frames, preprocessed_images, preprocessed_rgb, model, output_dir):
    # Iterar sobre cada par de imágenes y sus preprocesamientos
    for idx, (frame, preprocessed_yuv, preprocessed_rgb) in enumerate(zip(frames, preprocessed_images, preprocessed_rgb)):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        
        # Imagen YUV
        axes[0].imshow(frame)
        axes[0].set_title("YUV Image")
        axes[0].axis('off')
        
        # Predecir y mostrar las clases para la imagen YUV
        predictions_yuv = model.predict(np.expand_dims(preprocessed_yuv, axis=0))
        decoded_predictions_yuv = decode_predictions(predictions_yuv, top=5)[0]
        classes_yuv = [class_name for (_, class_name, _) in decoded_predictions_yuv]
        probs_yuv = [class_prob for (_, _, class_prob) in decoded_predictions_yuv]
        axes[0].text(-100, 40, f'Top 5 Classes (YUV):\n{classes_yuv[0]}: {probs_yuv[0]:.2f}\n{classes_yuv[1]}: {probs_yuv[1]:.2f}\n{classes_yuv[2]}: {probs_yuv[2]:.2f}\n{classes_yuv[3]}: {probs_yuv[3]:.2f}\n{classes_yuv[4]}: {probs_yuv[4]:.2f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # Imagen RGB
        axes[1].imshow(cv2.cvtColor(reescalar_matriz(preprocessed_rgb), cv2.COLOR_BGR2RGB))
        axes[1].set_title("RGB Image")
        axes[1].axis('off')
        
        # Predecir y mostrar las clases para la imagen RGB
        predictions_rgb = model.predict(np.expand_dims(preprocessed_rgb, axis=0))
        decoded_predictions_rgb = decode_predictions(predictions_rgb, top=5)[0]
        classes_rgb = [class_name for (_, class_name, _) in decoded_predictions_rgb]
        probs_rgb = [class_prob for (_, _, class_prob) in decoded_predictions_rgb]
        axes[1].text(-100, 40, f'Top 5 Classes (RGB):\n{classes_rgb[0]}: {probs_rgb[0]:.2f}\n{classes_rgb[1]}: {probs_rgb[1]:.2f}\n{classes_rgb[2]}: {probs_rgb[2]:.2f}\n{classes_rgb[3]}: {probs_rgb[3]:.2f}\n{classes_rgb[4]}: {probs_rgb[4]:.2f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # Guardar la figura en el directorio de salida
        fig.savefig(os.path.join(output_dir, f'image_{idx}.png'))
        plt.close(fig)

# Directorio de salida para guardar las imágenes
output_dir = "plots"

# Llamar a la función para plotear y guardar las imágenes
plot_and_save_images(frames_loaded, preprocessed_images, preprocessed_rgb, model, output_dir)
