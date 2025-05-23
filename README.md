# U_Net-OSMap_Skeletonization
Deep learning model to extract thin skeletons from thick, noisy road images generated from OpenStreetMap data. Uses U-Net and advanced techniques to handle skeletonization challenges. Includes data generation, training, and evaluation with Python scripts runnable on Colab.

Run procedures for Colab for both base and advanced model using Multitask learning through distance mapping:

1. First install needed libraries extract and copy text from **install.txt**
3. Create OSMap images of Oxford,OH and associated ground truth images (data folder)--> **%run make_data_v2.py**
4. Split data into 70% training/15% validation/15% test datasets in new directory (data folder)--> **%run data_split.py**

Continued procedures for base model (**Base_Model folder**):

1. Create Dataloaders of each dataset for Model use--> **%run create_Dataloaders.py**
2. Run function holding U-Net Base Model convolution blocks and forward pass --> **%run base_UNet_Model.py**
3. Run evaluate model function which holds functions for varied evaluation metrics including Test Loss, MSE, Dice Coefficient, and IoU **%run evaluate_Model.py**
4. Run model training using epochs=20 and combined Dice Loss/BCEwithLogits Loss --> **%run run_base_UNet_Model.py**
5. Run model evaluation metrics for Test Loss, Mean Squared Error, Dice Coefficient, and IoU stats --> **%run evaluate_node_metrics.py**
6. Run model plots for four sample images. Shows noisy image, ground truth, and predicted image. --> **%run matplot.py**

Continued procedures for advanced model (**Advanced_Model folder)**:
1. Create Dataloaders of each dataset for Model use--> **%run create_Dataloaders.py**
2. Run function holding U-Net Base Model convolution blocks and forward pass --> **%run advanced_UNet_Model.py**
3. 3. Run evaluate model function which holds functions for varied evaluation metrics including Test Loss, MSE, Dice Coefficient, and IoU **%run evaluate_Model.py**
4. Run model training using epochs=20 and combined Dice Loss/BCEwithLogits Loss --> **%run run_advanced_UNet_Model.py**
5. 5. Run model evaluation metrics for Test Loss, Mean Squared Error, Dice Coefficient, and IoU stats --> **%run evaluate_node_metrics.py**
6. Run model plots for four sample images. Shows noisy image, ground truth, and predicted image. --> **%run matplot.py**

Continue for both base and advanced model (**main folder** : 

1. To run tensorboard after to visualize use commands in **tensorboard.txt**
   
