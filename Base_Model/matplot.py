import torch
from base_UNet_Model import UNetStrided
import matplotlib.pyplot as plt
from create_Dataloaders import get_dataloaders  # or your actual dataloader module

output_dir = '/content/data/thinning/Oxford_split'  # or your actual data path
_, _, test_loader = get_dataloaders(output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNetStrided(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()
images, masks = next(iter(test_loader))
images = images.to(device)
outputs = model(images)
preds = torch.sigmoid(outputs) > 0.5  # Threshold

num_samples = min(5, images.shape[0])  # Show up to 5 samples

plt.figure(figsize=(12, 4 * num_samples))
for i in range(num_samples):
    # Input image
    plt.subplot(num_samples, 3, i * 3 + 1)
    plt.imshow(images[i, 0].cpu(), cmap='gray')
    plt.title('Input')
    plt.axis('off')

    # Ground truth mask
    plt.subplot(num_samples, 3, i * 3 + 2)
    plt.imshow(masks[i, 0].cpu(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    # Model prediction
    plt.subplot(num_samples, 3, i * 3 + 3)
    plt.imshow(preds[i, 0].cpu(), cmap='gray')
    plt.title('Prediction')
    plt.axis('off')

plt.tight_layout()
plt.show()
