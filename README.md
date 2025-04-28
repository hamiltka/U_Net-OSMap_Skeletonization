# U_Net-OSMap_Skeletonization-
Deep learning model to extract thin skeletons from thick, noisy road images generated from OpenStreetMap data. Uses U-Net and advanced techniques to handle skeletonization challenges. Includes data generation, training, and evaluation with Python scripts runnable on Colab.

Run procedures for Colab:

1. First install needed libraries extract and copy text from **install.txt**
2. Create OSMap images of Oxford,OH and associated ground truth images --> **%run make_data_v2.py**
3. Split data into 70% training/15% validation/15% test datasets in new directory --> **data_split.py**
4. Create Dataloaders of each dataset for Model use --> **create_Dataloaders.py**
5. 
