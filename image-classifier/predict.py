import argparse
from model_utils import load_checkpoint, predict
import json

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Predict flower name from an image with the probability of that name.')
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--arch', type=str, default='vgg11', help='Model architecture')

    args = parser.parse_args()

    # Load the checkpoint
    model = load_checkpoint(args.checkpoint, args.arch)

    # Make prediction
    probs, classes = predict(args.input, model, args.top_k, args.gpu)
    
    # Map classes to flower names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    class_names = [cat_to_name[class_] for class_ in classes]
    # Print results
    print("Probabilities:", probs)
    print("Classes:", class_names)

if __name__ == '__main__':
    main()
