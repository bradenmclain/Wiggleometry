import cv2
import os

# Function to crop images based on the selected bounding box
def crop_images(images, bbox):
    cropped_images = []
    x, y, w, h = bbox
    for img in images:
        cropped_img = img[y:y+h, x:x+w]
        cropped_images.append(cropped_img)
    return cropped_images

# Load images from a directory and return their filenames and image data
def load_images_from_directory(directory):
    image_list = []
    filenames = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if img is not None:
            image_list.append(img)
            filenames.append(filename)
    return filenames, image_list

# Directory containing the images
image_directory = '/home/delta/Wiggleometry/data/images'  # Replace with your directory path

# Load all images and their filenames
filenames, images = load_images_from_directory(image_directory)
if len(images) == 0:
    print("No images found in the directory.")
    exit()

# Display the first image and allow the user to select the bounding box
first_image = images[1].copy()
bbox = cv2.selectROI("Select Region", first_image, fromCenter=False, showCrosshair=True)

# If a bounding box was selected
if bbox != (0, 0, 0, 0):
    # Crop all images based on the selected bounding box
    cropped_images = crop_images(images, bbox)

    # Save the cropped images using their original names with '_cropped' appended
    output_dir = 'cropped_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, cropped_img in enumerate(cropped_images):
        original_filename = filenames[i]
        filename_without_ext, ext = os.path.splitext(original_filename)
        output_path = os.path.join(output_dir, f'{filename_without_ext}_cropped{ext}')
        cv2.imwrite(output_path, cropped_img)
        print(f"Saved: {output_path}")
else:
    print("No bounding box selected.")

# Release OpenCV windows
cv2.destroyAllWindows()
