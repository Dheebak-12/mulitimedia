import os
os.chdir(os.path.dirname(__file__))
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")      # to avoid Tkinter crash
import matplotlib.pyplot as plt



# ---------- LOAD IMAGES ----------
# Replace with your image path
img_low = cv2.imread("low_light.jpg", cv2.IMREAD_GRAYSCALE)
img_bright = cv2.imread("bright_light.jpg", cv2.IMREAD_GRAYSCALE)

# Ensure image loaded
if img_low is None or img_bright is None:
    raise ValueError("Error: Could not load one or both images.")

# ---------- FUNCTION TO COMPUTE BIT PLANES ----------
def compute_bitplanes(image):
    bitplanes = []
    for bit in range(8):
        plane = (image >> bit) & 1
        bitplanes.append((plane * 255).astype("uint8"))
    return bitplanes

# ---------- COMPUTE BIT PLANES ----------
bitplanes_low = compute_bitplanes(img_low)
bitplanes_bright = compute_bitplanes(img_bright)

# ---------- RECONSTRUCT USING LOWEST 3 BIT PLANES ----------
def reconstruct_from_low_bits(bitplanes):
    # bits 0, 1, 2
    recon = np.zeros_like(bitplanes[0], dtype=np.uint8)
    for bit in range(3):
        recon += ((bitplanes[bit]//255) << bit).astype(np.uint8)
    return recon

recon_low = reconstruct_from_low_bits(bitplanes_low)
recon_bright = reconstruct_from_low_bits(bitplanes_bright)

# ---------- DIFFERENCE BETWEEN ORIGINAL AND RECONSTRUCTED ----------
diff_low = cv2.absdiff(img_low, recon_low)
diff_bright = cv2.absdiff(img_bright, recon_bright)

# ---------- DISPLAY RESULTS ----------
titles = [
    "Original Low-light", "Reconstructed (3 lowest bits)", "Difference",
    "Original Bright-light", "Reconstructed (3 lowest bits)", "Difference"
]

images = [
    img_low, recon_low, diff_low,
    img_bright, recon_bright, diff_bright
]

plt.figure(figsize=(12, 8))

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("output.png")

