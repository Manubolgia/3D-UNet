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
    
    return parser.parse_args()

def setup_dist():
    """
    Setup for a single GPU environment.
    """
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    else:
        raise RuntimeError("CUDA is not available. A GPU is required for this setup.")

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

    criterion = CrossEntropyLoss()#weight=torch.Tensor(args.bce_weights)
    if args.train_cuda and torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = Adam(params=model.parameters())

    min_valid_loss = float('inf')
    
    # CSV logging setup
    if not os.path.exists(f'checkpoints/{args.resolution}_{args.scenario}'):
                os.makedirs(f'checkpoints/{args.resolution}_{args.scenario}')
    csv_file = f'checkpoints/{args.resolution}_{args.scenario}/training_metrics.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 
                             'Validation Precision', 'Validation Recall', 
                             'Validation Dice', 'Validation IoU'])


    for epoch in range(args.training_epoch):
        train_loss = 0.0
        model.train()
        for data in train_dataloader:
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

        valid_loss = 0.0
        all_precision, all_recall, all_dice, all_iou = [], [], [], []
        model.eval()
        with torch.no_grad():
            for data in val_dataloader:
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
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_valid_loss = valid_loss / len(val_dataloader)
        avg_precision = np.mean(all_precision)
        avg_recall = np.mean(all_recall)
        avg_dice = np.mean(all_dice)
        avg_iou = np.mean(all_iou)

        print(f'Epoch {epoch+1} \t\t Training Loss: {avg_train_loss} \t\t Validation Loss: {avg_valid_loss}')
        print(f'Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}')
        
        # Save metrics to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_valid_loss, avg_precision, avg_recall, avg_dice, avg_iou])

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            if not os.path.exists(f'checkpoints/{args.resolution}_{args.scenario}'):
                os.makedirs(f'checkpoints/{args.resolution}_{args.scenario}')
            torch.save(model.state_dict(), f'checkpoints/{args.resolution}_{args.scenario}/epoch{epoch}_valLoss{min_valid_loss:.6f}.pth')

if __name__ == '__main__':
    main()
