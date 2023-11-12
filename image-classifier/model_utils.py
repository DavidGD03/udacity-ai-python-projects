import torch
from torchvision import models
from torch import nn
from torch import optim
from collections import OrderedDict
from PIL import Image
import numpy as np

def build_model(arch='vgg11', hidden_units=512):
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier
    model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    return model

def train_model(model, train_loader, valid_loader, learning_rate=0.003, epochs=1, use_gpu=False):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    device = torch.device("cuda" if use_gpu else "cpu")
    steps = 0
    running_loss = 0
    print_every = 5
    criterion = nn.NLLLoss()
    model.to(device);
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
    return optimizer

def save_checkpoint(model, train_data, optimizer, save_dir='checkpoints/'):
    # Attach class_to_idx to the model
    model.class_to_idx = train_data.class_to_idx

    # Define additional information to save in the checkpoint
    checkpoint = {
        'input_size': 25088,  
        'output_size': 102,  
        'hidden_layers': [each.out_features for each in model.classifier if hasattr(each, 'out_features')],
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict()
    }

    # Save the checkpoint to a file
    torch.save(checkpoint, save_dir + 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    # Load a pre-trained model of the same architecture used during training
    model = models.vgg11(pretrained=True)  
    for param in model.parameters():
            param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('0', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0])),
        ('1', nn.ReLU()),
        ('2', nn.Dropout(0.2)),
        ('3', nn.Linear(checkpoint['hidden_layers'][0], checkpoint['output_size'])),
        ('4', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    #Load the state dict
    state_dict=checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # Attach class_to_idx to the model
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    
    Args:
    image_path (str): Path to the image file
    
    Returns:
    np_image (numpy array): Processed image as a Numpy array
    '''
    
    # Open the image
    pil_image = Image.open(image_path)
    
    # Resize the image
    pil_image.thumbnail((256, 256))
    
    # Crop the center 224x224 portion of the image
    left_margin = (pil_image.width - 224) / 2
    bottom_margin = (pil_image.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert color channels to floats in the range [0, 1]
    np_image = np.array(pil_image) / 255.0
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions to have color channel as the first dimension
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk=5, use_gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.

    Args:
    image_path (str): Path to the image file
    model (PyTorch model): Trained deep learning model
    topk (int): Number of top most likely classes to return
    
    Returns:
    probs (list): List of topk probabilities
    classes (list): List of topk class labels
    '''
    
    # Set the model to evaluation mode
    model.eval()
    
    # Preprocess the image
    processed_image = process_image(image_path)
    
    # Convert Numpy array to PyTorch tensor
    image_tensor = torch.from_numpy(processed_image).float()
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Move the input tensor to the GPU, if available
    device = torch.device("cuda" if use_gpu else "cpu")
    image_tensor = image_tensor.to(device)
    
    # Move the model to the same device as the input tensor
    model.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model.forward(image_tensor)
    
    # Calculate probabilities
    probabilities = torch.exp(output)
    
    # Get the topk probabilities and class indices
    top_probs, top_indices = probabilities.topk(topk)
    
    # Convert indices to class labels
    class_to_idx = model.class_to_idx
    idx_to_class = {idx: class_ for class_, idx in class_to_idx.items()}
    
    # Convert PyTorch tensors to lists
    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()
    
    # Map indices to class labels
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_probs, top_classes

