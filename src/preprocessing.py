import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_svm_features(image_rgb: np.ndarray) -> np.ndarray:
    """Reproduce the notebook's 48 RGB histogram + 4 GLCM feature pipeline."""
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected a three-channel RGB image.")
    rgb = cv2.resize(image_rgb, (256, 256))
    hist_features = []
    for channel in range(3):
        hist = cv2.calcHist([rgb], [channel], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist.tolist())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    texture = [
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
    ]
    return np.asarray(hist_features + texture, dtype=np.float32)

def preprocess_cnn_image(image_rgb: np.ndarray) -> np.ndarray:
    """RGB resize to 256x256 and float scaling to [0, 1], as in the notebook."""
    import tensorflow as tf
    image = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()
