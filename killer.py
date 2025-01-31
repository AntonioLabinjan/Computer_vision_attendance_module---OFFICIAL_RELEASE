import os
import shutil

def organize_images(base_folder, extra_folder):
    if not os.path.exists(extra_folder):
        os.makedirs(extra_folder)

    # Iterate through each subfolder in the base folder
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        
        # Skip if it's not a directory
        if not os.path.isdir(subfolder_path):
            continue
        
        # Get the list of images in the subfolder
        images = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
        
        # If there are more than 10 images, move the excess to the extra folder
        if len(images) > 10:
            excess_images = images[10:]
            for image in excess_images:
                source_path = os.path.join(subfolder_path, image)
                destination_path = os.path.join(extra_folder, image)
                shutil.move(source_path, destination_path)
                print(f"Moved {image} to {extra_folder}")

# Base folder containing subfolders with images
base_folder = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/bigger_dataset"
# Folder to store excess images
extra_folder = os.path.join(base_folder, "extra")

organize_images(base_folder, extra_folder)
