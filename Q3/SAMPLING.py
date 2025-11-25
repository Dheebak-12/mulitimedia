# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
from PIL import Image
import numpy as np


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------
path = "img.jpg"

# Convert one copy to grayscale → for frequency sampling
img_gray = Image.open(path).convert("L")

# Convert another copy to color RGB → for spatial sampling
img_rgb = Image.open(path).convert("RGB")

# Convert both to numpy arrays
gray = np.array(img_gray)      # shape = (H, W)
rgb = np.array(img_rgb)        # shape = (H, W, 3)

print("Grayscale Resolution :", gray.shape)
print("RGB Resolution       :", rgb.shape)


# ---------------------------------------------------------
# FREQUENCY SAMPLING (Using FFT)
# ---------------------------------------------------------
def freq_sample(image, factor):
    """
    image : grayscale numpy array
    factor: (2,4,8,16) - controls how much frequency we keep

    This function:
    1) Converts image to frequency domain (FFT)
    2) Keeps only the low-frequency center region
    3) Converts back to spatial domain (inverse FFT)
    4) Normalizes image to uint8
    """

    # 1. FFT → frequency domain
    F = np.fft.fft2(image)
    F = np.fft.fftshift(F)   # shift low-freq to center
    H, W = F.shape

    # 2. Define how big the kept region is
    h_keep = max(1, H // factor)
    w_keep = max(1, W // factor)

    # Create an empty frequency image
    F_low = np.zeros_like(F)

    # 3. Compute center coordinates
    hs = (H // 2) - (h_keep // 2)
    he = hs + h_keep

    ws = (W // 2) - (w_keep // 2)
    we = ws + w_keep

    # Copy only the low-frequency center block
    F_low[hs:he, ws:we] = F[hs:he, ws:we]

    # 4. Inverse FFT → back to image
    img_back = np.fft.ifft2(np.fft.ifftshift(F_low))
    img_back = np.abs(img_back)  # magnitude only

    # 5. Normalize to uint8 (0–255)
    min_val, max_val = img_back.min(), img_back.max()
    if max_val - min_val < 1e-9:
        img_uint8 = img_back.astype(np.uint8)
    else:
        img_uint8 = ((img_back - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return img_uint8



# ---------------------------------------------------------
# SPATIAL SAMPLING (RGB)
# ---------------------------------------------------------
def spatial_sample(image, factor):
    """
    image : RGB numpy array
    factor: (2,4,8,16)
    Takes every 'factor'-th row and column.
    """

    # Pick every f-th pixel in both directions
    sampled = image[::factor, ::factor]

    return sampled.copy()



# ---------------------------------------------------------
# RUN SAMPLING FOR MULTIPLE FACTORS
# ---------------------------------------------------------
factors = [2, 4, 8, 16]

for f in factors:
    # -------------------
    # Frequency sampling
    # -------------------
    out_freq = freq_sample(gray, f)
    fname_freq = f"freq_1_{f}.png"
    Image.fromarray(out_freq).save(fname_freq)

    print(f"\nFrequency Sampling 1/{f}")
    print("Output Resolution:", out_freq.shape)
    print("Saved:", fname_freq)

    # -------------------
    # Spatial sampling
    # -------------------
    out_spat = spatial_sample(rgb, f)
    fname_spat = f"spatial_1_{f}.png"
    Image.fromarray(out_spat).save(fname_spat)

    print(f"Spatial Sampling 1/{f}")
    print("Output Resolution:", out_spat.shape)
    print("Saved:", fname_spat)


print("\nAll images saved in the current directory.")
