from PIL import Image
#import numpy as np

def grayscale(image):
    width, height = input.size

    # Create a new grayscale image
    quant_img = Image.new("L", (width, height))

    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            r, g, b = input.getpixel((x, y))
            gray = int((max(r, g, b) + min(r, g, b)) / 2)  # Lightness method
            quant_img.putpixel((x, y), gray)

    return quant_img    

# Load source image
input=Image.open("input.jpg")

#Grayscale
gray_img=grayscale(input)
gray_img.save("gray_lightness_forloop.jpg")

print("quantization completed")