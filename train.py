import torch
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from unet3d import UNet3D
import os
from dataset import get_dataloaders
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import numpy as np
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for medical image segmentation")
    
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--resolution', type=int, choices=[64, 128], required=True, help='Image resolution (64 or 128)')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--training_epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--train_cuda', action='store_true', help='Use CUDA for training if available')
    parser.add_argument('--bce_weights', type=float, nargs='+', default=[0.004, 0.996], help='Weights for binary cross-entropy loss')
    parser.add_argument('--scenario', type=str, default='1', choices=['1','2','3','4'], help='Scenario to train the model on')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    return parser.parse_args()

def setup_dist():
    """
    Setup for a single GPU environment.
    """
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    else:
        raise RuntimeError("CUDA is not available. A GPU is required for this setup.")

def calculate_class_weights(dataloader, num_classes):
    class_counts = torch.zeros(num_classes)
    
    for batch in dataloader:
        _, labels, _ = batch['image'], batch['label'], batch['affine']
        for label in labels:
            unique, counts = torch.unique(label, return_counts=True)
            class_counts[unique.long()] += counts

    total_counts = class_counts.sum()
    class_weights = total_counts / (num_classes * class_counts)
    
    return class_weights

def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data = data.long()

    # create one-hot label map
    label_map = data
    bs, _, d, h, w = label_map.size()
    nc = num_classes
    input_label = torch.FloatTensor(bs, nc, d, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)     
    return input_semantics

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    args = parse_args()

    model = UNet3D(in_channels=args.in_channels, num_classes=args.num_classes)

    setup_dist()
    if args.train_cuda and torch.cuda.is_available():
        model = model.cuda()

    train_dataloader, val_dataloader, _ = get_dataloaders(
        data_dir=args.data_dir,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        scenario=args.scenario
    )
    # Compute class weights based on the training data
    class_weights = calculate_class_weights(train_dataloader, args.num_classes)

    # Convert class weights to a tensor and move to GPU if necessary
    class_weights = torch.FloatTensor(class_weights)
    if args.train_cuda and torch.cuda.is_available():
        class_weights = class_weights.cuda()

    # Define the loss function with class weights
    criterion = CrossEntropyLoss(weight=class_weights)
    
    if args.train_cuda and torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = Adam(params=model.parameters())

    results_dir = f'Results/{args.resolution}_{args.scenario}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    log_file = os.path.join(results_dir, 'training.log')
    setup_logging(log_file)

    min_valid_loss = float('inf')
    no_improve_epochs = 0
    
    # CSV logging setup
    csv_file = os.path.join(results_dir, 'training_metrics.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 
                             'Validation Precision', 'Validation Recall', 
                             'Validation Dice', 'Validation IoU'])

    for epoch in range(args.training_epoch):
        train_loss = 0.0
        model.train()

        logging.info(f"Starting Epoch {epoch+1}")

        for batch_idx, data in enumerate(train_dataloader):
            image, ground_truth, affine = data['image'], data['label'], data['affine']
            ground_truth = preprocess_input(ground_truth, args.num_classes)
            ground_truth = torch.argmax(ground_truth, dim=1)
            if args.train_cuda and torch.cuda.is_available():
                image = image.cuda()
                ground_truth = ground_truth.cuda()

            optimizer.zero_grad()
            target = model(image)
            loss = criterion(target, ground_truth)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            #if batch_idx % 10 == 0:  # Log every 10 batches
            #    logging.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}')

        valid_loss = 0.0
        all_precision, all_recall, all_dice, all_iou = [], [], [], []
        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                image, ground_truth, affine = data['image'], data['label'], data['affine']
                ground_truth = preprocess_input(ground_truth, args.num_classes)
                ground_truth = torch.argmax(ground_truth, dim=1)
                if args.train_cuda and torch.cuda.is_available():
                    image = image.cuda()
                    ground_truth = ground_truth.cuda()

                target = model(image)
                loss = criterion(target, ground_truth)
                valid_loss += loss.item()

                # Compute additional metrics
                target_np = torch.argmax(target, dim=1).cpu().numpy()
                ground_truth_np = ground_truth.cpu().numpy()

                precision = precision_score(ground_truth_np.flatten(), target_np.flatten(), average='weighted', labels=np.arange(1, args.num_classes), zero_division=0)
                recall = recall_score(ground_truth_np.flatten(), target_np.flatten(), average='weighted', labels=np.arange(1, args.num_classes), zero_division=0)
                dice = f1_score(ground_truth_np.flatten(), target_np.flatten(), average='weighted', labels=np.arange(1, args.num_classes), zero_division=0)
                iou = jaccard_score(ground_truth_np.flatten(), target_np.flatten(), average='weighted', labels=np.arange(1, args.num_classes), zero_division=0)

                all_precision.append(precision)
                all_recall.append(recall)
                all_dice.append(dice)
                all_iou.append(iou)

                #if batch_idx % 10 == 0:  # Log every 10 batches
                #    logging.info(f'Epoch {epoch+1}, Validation Batch {batch_idx}, Loss: {loss.item()}')

        avg_train_loss = train_loss / len(train_dataloader)
        avg_valid_loss = valid_loss / len(val_dataloader)
        avg_precision = np.mean(all_precision)
        avg_recall = np.mean(all_recall)
        avg_dice = np.mean(all_dice)
        avg_iou = np.mean(all_iou)

        logging.info(f'Epoch {epoch+1} Completed')
        logging.info(f'Training Loss: {avg_train_loss:.4f}')
        logging.info(f'Validation Loss: {avg_valid_loss:.4f}')
        logging.info(f'Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}')
        
        # Save metrics to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_valid_loss, avg_precision, avg_recall, avg_dice, avg_iou])

        # Early stopping logic
        if avg_valid_loss < min_valid_loss:
            logging.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{avg_valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = avg_valid_loss
            no_improve_epochs = 0
            checkpoints_dir = os.path.join(results_dir, 'checkpoints')
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'epoch{epoch}_valLoss{min_valid_loss:.6f}.pth'))
        else:
            no_improve_epochs += 1
            logging.info(f'No improvement in validation loss for {no_improve_epochs} epochs.')

        if no_improve_epochs >= args.patience:
            logging.info(f'Early stopping triggered after {no_improve_epochs} epochs without improvement.')
            break

if __name__ == '__main__':
    main()
