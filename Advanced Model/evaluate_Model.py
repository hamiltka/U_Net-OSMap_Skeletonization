import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def mse_distance_transform(pred, gt):
    dt_gt = distance_transform_edt(1 - gt)
    return np.mean((pred - dt_gt) ** 2)

def iou_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 0

def dice_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

def evaluate_model(model, test_loader, device, combined_loss):
    model.eval()
    test_loss = 0.0
    mse_total = 0.0
    iou_total = 0.0
    dice_total = 0.0
    n_samples = 0
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (images, masks, [optional extra outputs])
            if len(batch) == 2:
                images, masks = batch
            else:
                images, masks = batch[:2]
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            # If model returns multiple outputs, use the first for segmentation
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = combined_loss(outputs, masks)
            test_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(images.size(0)):
                pred_bin = preds[i, 0]
                mask_bin = masks_np[i, 0]
                mse_total += mse_distance_transform(pred_bin, mask_bin)
                iou_total += iou_score(pred_bin, mask_bin)
                dice_total += dice_score(pred_bin, mask_bin)
                n_samples += 1

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_mse = mse_total / n_samples
    avg_iou = iou_total / n_samples
    avg_dice = dice_total / n_samples

    print(f"\n--- Test Set Metrics ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"MSE (Distance Transform): {avg_mse:.4f}")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Dice Coefficient: {avg_dice:.4f}")

    return {
        "test_loss": avg_test_loss,
        "mse": avg_mse,
        "iou": avg_iou,
        "dice": avg_dice
    }
