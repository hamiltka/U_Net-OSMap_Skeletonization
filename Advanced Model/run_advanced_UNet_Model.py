import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from advanced_UNet_Model import UNetMultiTask 
from evaluate_Model import evaluate_model  # Ensure this is updated with node metrics
from create_Dataloaders import get_dataloaders

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = UNetMultiTask(in_channels=1, out_channels=1).to(device)

# DataLoaders
output_dir = '/content/data/thinning/Oxford_split'
train_loader, val_loader, test_loader = get_dataloaders(output_dir)

# Losses
def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    target = target.float()
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

bce = torch.nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return bce(pred, target) + dice_loss(pred, target)

loss_distance = torch.nn.MSELoss()
alpha = 1.0

optimizer = optim.Adam(model.parameters(), lr=1e-3)
writer = SummaryWriter(log_dir='runs/experiment1')
total_start = time.time()
num_epochs = 20

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    
    # Training
    for batch_idx, (images, skel_masks, dist_maps) in enumerate(train_loader):
        images = images.to(device)
        skel_masks = skel_masks.to(device)
        dist_maps = dist_maps.to(device)

        optimizer.zero_grad()
        pred_skel, pred_dist = model(images)
        loss_skel = combined_loss(pred_skel, skel_masks)
        loss_dist = loss_distance(torch.sigmoid(pred_dist), dist_maps)
        loss = loss_skel + alpha * loss_dist
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    avg_loss = running_loss / len(train_loader.dataset)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"  --> Training Loss (epoch avg): {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, skel_masks, dist_maps) in enumerate(val_loader):
            images = images.to(device)
            skel_masks = skel_masks.to(device)
            dist_maps = dist_maps.to(device)
            pred_skel, pred_dist = model(images)
            loss_skel = combined_loss(pred_skel, skel_masks)
            loss_dist = loss_distance(torch.sigmoid(pred_dist), dist_maps)
            loss = loss_skel + alpha * loss_dist
            val_loss += loss.item() * images.size(0)
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    print(f"  --> Validation Loss (epoch avg): {avg_val_loss:.4f}")
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds")

total_time = time.time() - total_start
print(f"\nTotal training time: {total_time:.2f} seconds")
writer.close()

# Save model
torch.save(model.state_dict(), 'model.pth')
print("Model weights saved as model.pth")

# Final Evaluation
print("\nRunning Final Test Evaluation...")
results = evaluate_model(model, test_loader, device, combined_loss)

