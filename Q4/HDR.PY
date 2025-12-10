import os
os.chdir(os.path.dirname(__file__))
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

images = []
exposure_times = []
files = sorted(glob.glob("data/*.JPG"))   

for f in files:
    img = cv2.imread(f)
    if img is None:
        print("Could not read:", f)
        continue

    images.append(img)

    filename = os.path.basename(f)
    name_wo_ext = os.path.splitext(filename)[0]
    exposure_str = name_wo_ext.split("_")[-1]

    try:
        t = float(exposure_str)
    except:
        print("ERROR extracting exposure from:", filename)
        continue

    exposure_times.append(t)

images = np.array(images)  
exposure_times = np.array(exposure_times, dtype=np.float32)

print("Loaded", len(images), "images with exposures:", exposure_times)

calibrate = cv2.createCalibrateDebevec()
response = calibrate.process(images, exposure_times)
print("CRF estimated.")

merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(images, exposure_times, response)
print("HDR Radiance Map computed.")

cv2.imwrite("output_hdr.hdr", hdr)
print("Saved HDR radiance map: output_hdr.hdr")

tonemap = cv2.createTonemapReinhard(gamma=1.4, intensity=0)
ldr = tonemap.process(hdr)

ldr = np.nan_to_num(ldr, nan=0.0, posinf=1.0, neginf=0.0)
ldr = np.clip(ldr * 255, 0, 255).astype("uint8")
cv2.imwrite("output_tonemapped.jpg", ldr)

print("Saved tone-mapped image: output_tonemapped.jpg")

plt.imshow(cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB))
plt.title("Tone-Mapped HDR")
plt.axis('off')
plt.show()