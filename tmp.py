from PIL import Image, ImageOps

def add_white_border(input_path, output_path, border_px):
    # Open the image
    img = Image.open(input_path)
    # Add a white border of border_px on all sides
    bordered_img = ImageOps.expand(img, border=border_px, fill='white')
    # Save the resulting image
    bordered_img.save(output_path)

# Example usage:
if __name__ == '__main__':
    add_white_border("assets/track-v2.jpg", "assets/track-v2.jpg", 20)
