from PIL import Image
def apply_median_cut(source_file, result_file, max_colors=16):
    """
    Reduce the total number of colors in an image
    using PIL's Median Cut algorithm.
    """

    # Load the input picture (force RGB mode)
    original = Image.open(source_file).convert("RGB")

    # Apply quantization
    # method = 0 corresponds to Median Cut
    reduced = original.quantize(colors=max_colors, method=0)

    # Convert back to RGB so RGB-based formats save correctly
    final_img = reduced.convert("RGB")

    # Store the processed output image
    final_img.save(result_file)
    print("Image successfully saved to:", result_file)


# Example call
apply_median_cut("input.jpg", "output_median_cut.jpg", max_colors=10)
