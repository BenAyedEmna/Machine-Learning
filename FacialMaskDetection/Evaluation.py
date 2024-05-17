import numpy as np
import os
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img

def predict(model, img):
    x = np.expand_dims(img, axis=0)  # Reshape for model input
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]  # Return raw probabilities for flexibility

def evaluate_class(model, img_path, class_index, threshold):
    true_positives = 0
    total_images = 0
    false_detections = []

    for img_name in sorted(os.listdir(img_path)):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue

        filepath = os.path.join(img_path, img_name)
        img = load_img(filepath, target_size=(64, 64))  # Adjust target size if needed

        preds = predict(model, img)
        total_images += 1

        if preds[class_index] >= threshold:
            true_positives += 1
        else:
            false_detections.append(filepath)

    return true_positives, total_images, false_detections

def calculate_accuracy(true_positives, total_images):
    return (true_positives / total_images) * 100 if total_images > 0 else 0.0

def print_and_save_results(class_name, accuracy, false_detections, output_filepath, total_images):
    print(f"Accuracy for '{class_name}' class: {accuracy:.2f}%")
    print(f"Total Images: {len(false_detections) + total_images}")
    print(f"False Detections: {len(false_detections)}\n")

    with open(output_filepath, "w") as f:
        for filepath in false_detections:
            f.write(filepath + "\n")


def main():
    model_path = 'facemask4.h5'
    model = load_model(model_path)

    # Evaluate for "face" class
    true_positives_face, total_images_face, false_detections_face = evaluate_class(model, 'D:/Users/henta/Bureau/2 ing/deep learning/tp/FaceMaskDataset/FaceMaskDataset/test224/face', 0, 0.5)
    accuracy_face = calculate_accuracy(true_positives_face, total_images_face)

    # Evaluate for "maskface" class
    true_positives_maskface, total_images_maskface, false_detections_maskface = evaluate_class(model, 'D:/Users/henta/Bureau/2 ing/deep learning/tp/FaceMaskDataset/FaceMaskDataset/test224/maskface', 1, 0.5)
    accuracy_maskface = calculate_accuracy(true_positives_maskface, total_images_maskface)

    overall_accuracy = (accuracy_face + accuracy_maskface) / 2

    # Print and save results for "face" class
    print_and_save_results("face", accuracy_face, false_detections_face, "falsedetectionFace.txt", total_images_face)

    # Print and save results for "maskface" class
    print_and_save_results("maskface", accuracy_maskface, false_detections_maskface, "falsedetectionMaskface.txt",
                           total_images_maskface)

    print(f"Overall accuracy: {overall_accuracy:.2f}%")


if __name__ == "__main__":
    main()
