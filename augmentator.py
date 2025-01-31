import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Define augmentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.Resize(224, 224),  # Ensuring uniform image size
])

# Paths
BASE_DIR = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/known_faces"

# Iterate through each class folder
for class_folder in tqdm(os.listdir(BASE_DIR), desc="Processing Classes"):
    class_path = os.path.join(BASE_DIR, class_folder)
    train_path = os.path.join(class_path, "train")
    
    if not os.path.exists(train_path):  # Skip if no train folder
        print(f"No train folder found in {class_folder}")
        continue

    # Create augmented folder for each class
    augmented_path = os.path.join(class_path, "augmented_train")
    os.makedirs(augmented_path, exist_ok=True)

    # Iterate through train images and apply augmentations
    for image_file in os.listdir(train_path):
        image_path = os.path.join(train_path, image_file)
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        # Read and augment the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        for i in range(3):  # Generate 3 augmented versions per image
            augmented = augmentations(image=image)["image"]
            augmented_filename = os.path.join(augmented_path, f"{os.path.splitext(image_file)[0]}_aug{i}.jpg")
            cv2.imwrite(augmented_filename, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    print(f"Augmentations applied to train folder of {class_folder}")
