import os
import argparse
from PIL import Image, ImageChops

def crop_whitespace(img, background_color=(255, 255, 255), margin=0):
    """Crop whitespace from around an image with a given margin."""
    # Convert image to RGBA if not already in that mode
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Create a mask for the background color
    background = Image.new('RGBA', img.size, background_color)
    diff = ImageChops.difference(img, background)
    bbox = diff.getbbox()

    if bbox:
        # Crop the image to the bounding box with an added margin
        left, upper, right, lower = bbox
        left -= margin
        upper -= margin
        right += margin
        lower += margin

        # Ensure the coordinates are within bounds
        left = max(left, 0)
        upper = max(upper, 0)
        right = min(right, img.width)
        lower = min(lower, img.height)

        return img.crop((left, upper, right, lower))
    
    return img  # Return original if no cropping is needed

def join_images(folder):
    frame_files = {}
    time_files = {}

    for filename in os.listdir(folder):
        if filename.startswith("frame_time_") and filename.endswith(".png"):
            key = filename.replace("frame_time_", "").replace(".png", "")
            time_files[key] = filename
        elif filename.startswith("frame_") and filename.endswith(".png") and not filename.startswith("frame_time_"):
            key = filename.replace("frame_", "").replace(".png", "")
            frame_files[key] = filename

    common_keys = set(frame_files.keys()) & set(time_files.keys())

    for key in sorted(common_keys):
        path_time = os.path.join(folder, time_files[key])
        path_frame = os.path.join(folder, frame_files[key])

        img_time = crop_whitespace(Image.open(path_time), margin=40)
        img_frame = crop_whitespace(Image.open(path_frame), margin=40)

        # Align heights after cropping (optional)
        max_height = max(img_time.height, img_frame.height)
        def pad_height(img):
            if img.height == max_height:
                return img
            new_img = Image.new('RGB', (img.width, max_height), (255, 255, 255))
            new_img.paste(img, (0, (max_height - img.height) // 2))
            return new_img

        img_time = pad_height(img_time)
        img_frame = pad_height(img_frame)

        # Join images
        new_width = img_time.width + img_frame.width
        joined_img = Image.new('RGB', (new_width, max_height), (255, 255, 255))
        joined_img.paste(img_frame, (0, 0))
        joined_img.paste(img_time, (img_frame.width+40, 0))

        out_path = os.path.join(folder, f"joined_{key}.png")
        joined_img.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join paired frame and frame_time images without whitespace")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing the PNG files")
    args = parser.parse_args()

    join_images(args.folder)
