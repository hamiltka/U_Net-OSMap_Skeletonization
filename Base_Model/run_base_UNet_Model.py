import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from base_UNet_Model import UNetStrided

from create_Dataloaders import get_dataloaders

output_dir = '/content/data/thinning/Oxford_split'
train_loader, val_loader, test_loader = get_dataloaders(output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetStrided(in_channels=1, out_channels=1).to(device)

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

optimizer = optim.Adam(model.parameters(), lr=1e-3)

writer = SummaryWriter(log_dir='runs/experiment1')
total_start = time.time()

num_epochs = 20

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0

    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        # Print batch progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Train Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader.dataset)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"  --> Training Loss (epoch avg): {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            val_loss += loss.item() * images.size(0)

            # Print batch progress every 5 batches
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(val_loader):
                print(f"  Val Batch [{batch_idx+1}/{len(val_loader)}] "
                      f"Loss: {loss.item():.4f}")

    avg_val_loss = val_loss / len(val_loader.dataset)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    print(f"  --> Validation Loss (epoch avg): {avg_val_loss:.4f}")

    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds.")

total_time = time.time() - total_start
print(f"\nTotal training time: {total_time:.2f} seconds")
writer.close()

# Save the trained model weights
torch.save(model.state_dict(), 'model.pth')
print("Model weights saved as model.pth")
