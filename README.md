# Info
In this branch a sequence of segmentation model and cyst classifier is implemented in order to train the two models together. Based on two-models-sequence-unfold branch.
Patch extraction is perfomed using unfold method in order to gain performances and keep gradient calculations correct, without interruptions.
Caranet (no multi-scale) in sequence with classifier. 
Images are resized by a 0.75 factor.

Dataset used: https://paperswithcode.com/dataset/digestpath 

Obtained from: https://github.com/bupt-ai-cz/CAC-UNet-DigestPath2019/blob/main/README.md 

DigestPath dataset is divided in two sections:
-  positives: 250 images with relative segmenation masks
- negatives: images without segmentation masks because it should be all black (no target objects to segment in these images)

For the moment only positives will be used to train and test model, moreover the dataset structure has been edited:
In the version used positive images and relative masks have been separated in two folders (images and masks respectively) and masks files have been bulk renamed in order to have images and relative mask with the same name but in different folders. In the original dataset structure images and names are in a single folder and masks have suffix _mask. 