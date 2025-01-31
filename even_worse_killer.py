import os
import random
import shutil

def split_dataset(base_folder):
    for class_folder in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_folder)
        
        # Skip if it's not a directory
        if not os.path.isdir(class_path):
            continue
        
        # Create train and val directories inside each class folder
        train_path = os.path.join(class_path, "train")
        val_path = os.path.join(class_path, "val")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        
        # Get all images in the class folder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        # Shuffle the images to ensure random distribution
        random.shuffle(images)
        
        # Split images into train and val sets
        train_images = images[:8]
        val_images = images[8:10]
        
        # Move images to respective folders
        for image in train_images:
            shutil.move(os.path.join(class_path, image), os.path.join(train_path, image))
        for image in val_images:
            shutil.move(os.path.join(class_path, image), os.path.join(val_path, image))
        
        print(f"Processed class: {class_folder}")

# Path to the folder containing subfolders named after classes
base_folder = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/known_faces"

split_dataset(base_folder)
