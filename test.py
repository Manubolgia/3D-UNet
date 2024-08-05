import torch
import argparse
import os
import numpy as np
import nibabel as nib
from unet3d import UNet3D
from torch.nn import CrossEntropyLoss
from dataset import get_dataloaders
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Test 3D U-Net model for medical image segmentation")

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--resolution', type=int, choices=[64, 128], required=True, help='Image resolution (64 or 128)')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--train_cuda', action='store_true', help='Use CUDA for testing if available')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')

    return parser.parse_args()

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

def save_nifti(data, filepath, affine):
    img = nib.Nifti1Image(data.astype(np.int32), affine=affine.squeeze())
    nib.save(img, filepath)

def main():
    args = parse_args()

    model = UNet3D(in_channels=args.in_channels, num_classes=args.num_classes)

    if args.train_cuda and torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    _, _, test_dataloader = get_dataloaders(
        data_dir=args.data_dir,
        resolution=args.resolution,
        test_batch_size=args.test_batch_size,
    )

    if not os.path.exists('results'):
        os.makedirs('results')

    total_loss = 0.0
    criterion = CrossEntropyLoss()
    if args.train_cuda and torch.cuda.is_available():
        criterion = criterion.cuda()

    all_precision, all_recall, all_dice, all_iou = [], [], [], []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            image, ground_truth, affine = data['image'], data['label'], data['affine']
            ground_truth = preprocess_input(ground_truth, args.num_classes)
            ground_truth = torch.argmax(ground_truth, dim=1)

            if args.train_cuda and torch.cuda.is_available():
                image = image.cuda()
                ground_truth = ground_truth.cuda()

            target = model(image)
            loss = criterion(target, ground_truth)
            total_loss += loss.item()

            target_np = torch.argmax(target, dim=1).cpu().numpy()
            ground_truth_np = ground_truth.cpu().numpy()

            precision = precision_score(ground_truth_np.flatten(), target_np.flatten(), average='weighted', labels=np.arange(1, args.num_classes), zero_division=0)
            recall = recall_score(ground_truth_np.flatten(), target_np.flatten(), average='weighted', labels=np.arange(1, args.num_classes))
            dice = f1_score(ground_truth_np.flatten(), target_np.flatten(), average='weighted', labels=np.arange(1, args.num_classes))
            iou = jaccard_score(ground_truth_np.flatten(), target_np.flatten(), average='weighted', labels=np.arange(1, args.num_classes))


            all_precision.append(precision)
            all_recall.append(recall)
            all_dice.append(dice)
            all_iou.append(iou)

            # Save segmentation as NIfTI
            save_nifti(target_np[0], os.path.join('results', f'segment_{i}.nii.gz'), affine)

    avg_loss = total_loss / len(test_dataloader)
    avg_precision = np.mean(all_precision)
    std_precision = np.std(all_precision)

    avg_recall = np.mean(all_recall)
    std_recall = np.std(all_recall)

    avg_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)

    avg_iou = np.mean(all_iou)
    std_iou = np.std(all_iou)

    # Report the results with standard deviation
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Precision: {avg_precision:.4f} ± {std_precision:.4f}')
    print(f'Recall: {avg_recall:.4f} ± {std_recall:.4f}')
    print(f'Dice Coefficient: {avg_dice:.4f} ± {std_dice:.4f}')
    print(f'Mean IoU: {avg_iou:.4f} ± {std_iou:.4f}')

if __name__ == '__main__':
    main()
