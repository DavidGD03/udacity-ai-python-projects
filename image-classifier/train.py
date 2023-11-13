import argparse
from model_utils import build_model, train_model, save_checkpoint
from data_utils import load_data

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg11', help='Architecture (e.g., "vgg11")')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Load and preprocess data
    train_loader, valid_loader, test_loader, train_data = load_data(args.data_directory)

    # Build the model
    model = build_model(args.arch, args.hidden_units)

    # Train the model
    optimizer = train_model(model, train_loader, valid_loader, args.learning_rate, args.epochs, args.gpu)

    # Save the checkpoint
    save_checkpoint(model, train_data, optimizer, args.arch, args.save_dir)

if __name__ == '__main__':
    main()
