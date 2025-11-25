from PIL import Image

def apply_octree_quantization(source_file, result_file, max_colors):
    """
    Reduce the total number of colors in an image
    using the Octree Color Quantization algorithm.
    """

    # Load image (force RGB mode)
    original = Image.open(source_file).convert("RGB")

    # Octree quantization: method=2 in Pillow
    reduced = original.quantize(colors=max_colors, method=2)

    # Convert back to RGB so we can save properly
    final_img = reduced.convert("RGB")

    # Save result
    final_img.save(result_file)
    print("Image successfully saved to:", result_file)


max_colors = int(input("Enter maximum colors: "))
apply_octree_quantization("input.jpg", "output_octree.jpg", max_colors)
