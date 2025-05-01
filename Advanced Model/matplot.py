import torch
from advanced_UNet_Model import UNetMultiTask
import matplotlib.pyplot as plt
from create_Dataloaders import get_dataloaders  # or your actual dataloader module

output_dir = '/content/data/thinning/Oxford_split'  # or your actual data path
_, _, test_loader = get_dataloaders(output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNetMultiTask(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()
images, skel_masks, dist_maps = next(iter(test_loader))
images = images.to(device)
with torch.no_grad():
    pred_skel, pred_dist = model(images)
    pred_skel = torch.sigmoid(pred_skel)
    pred_dist = torch.sigmoid(pred_dist)  # if you want [0,1] output

num_samples = min(5, images.shape[0])
for i in range(num_samples):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(images[i, 0].cpu(), cmap='gray')
    plt.title('Input')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(skel_masks[i, 0].cpu(), cmap='gray')
    plt.title('Skeleton GT')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(pred_skel[i, 0].cpu() > 0.5, cmap='gray')
    plt.title('Skeleton Pred')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(pred_dist[i, 0].cpu(), cmap='jet')
    plt.title('Distance Pred')
    plt.axis('off')
    plt.show()