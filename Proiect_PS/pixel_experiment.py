import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

video_path = "football.mp4"
x, y = 100, 100
num_frames = 100
sigma_2d = 1.0
sigma_3d = (1.0, 1.0, 1.0)

cap = cv2.VideoCapture(video_path)
frames = []
count = 0
while count < num_frames:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)
    count += 1
cap.release()

frames = np.array(frames, dtype=np.float32)

# Intensitate pixel original
pixel_vals_original = frames[:, y, x]

# Filtrare 2D (pe cadru)
frames_2d = np.array([cv2.GaussianBlur(f, (3,3), sigma_2d) for f in frames])
pixel_vals_2d = frames_2d[:, y, x]

# Filtrare 3D (spatio-temporala)
frames_3d = gaussian_filter(frames, sigma=sigma_3d)
pixel_vals_3d = frames_3d[:, y, x]
plt.figure(figsize=(10,5))
plt.plot(pixel_vals_original, label='Original (zgomot)')
plt.plot(pixel_vals_2d, label='Filtrare 2D (clickering)')
plt.plot(pixel_vals_3d, label='Filtrare 3D (stabil)')
plt.xlabel("Cadru")
plt.ylabel("Intensitate pixel")
plt.legend()
plt.title("Stabilitate temporală a pixelului")
plt.show()
