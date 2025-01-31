from transformers import CLIPModel, CLIPProcessor
import os
import torch

# ako sve sjebemo, tornat iz githuba!!!!!
def load_clip_model(weights_path, dataset_path):
    """
    Loads the CLIP model and processor, initializes the classifier,
    and loads weights if the dataset contains known classes.
    """
    print(f"Looking for dataset in: {dataset_path}")
    
    # Check if dataset path exists, if not, create it
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist. Creating: {dataset_path}")
        os.makedirs(dataset_path)  # Create the directory if it doesn't exist
    
    # Determine number of classes from the dataset
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # If there are no classes, we proceed with a default setting
    if not class_dirs:
        print("Warning: No class directories found in dataset path. Proceeding with a default class setup.")
        num_classes = 1  # Default class setup (you can change this as per your needs)
    else:
        print(f"Classes found: {class_dirs}")
        num_classes = len(class_dirs)
    
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Initialize the classifier based on the number of classes
    classifier = torch.nn.Linear(model.config.projection_dim, num_classes)
    
    print(f"Looking for weights in: {weights_path}")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path)
        if state_dict['weight'].shape == classifier.weight.shape and state_dict['bias'].shape == classifier.bias.shape:
            classifier.load_state_dict(state_dict)
            print("Loaded classifier weights successfully.")
        else:
            print("Warning: Saved weights do not match the current classifier dimensions. Initializing with new weights.")
    else:
        print("No weights found. Initializing classifier with random weights.")
    
    model.eval()
    classifier.eval()
    return model, processor, classifier

if __name__ == "__main__":
    # Define paths for the Docker container
    weights_path = "/app/fine_tuned_classifier.pth"  # Path to weights in the container
    dataset_path = "/app/known_faces"  # Path to dataset in the container

    # Load the model and processor
    model, processor, classifier = load_clip_model(weights_path, dataset_path)
