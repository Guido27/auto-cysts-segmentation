# Final code version is in this branch!

## Info
In this branch a sequence of segmentation model and cyst classifier is implemented in order to train the two models together.
Patch extraction is perfomed using **unfold** method in order to gain performances in terms of both time and gradient calculations correctness, without interruptions.

- Caranet (no multi-scale) in sequence with Res2Net classifier. 
- Images and masks are resized by a 0.75 factor. 
- Augmentations are applied.

## Specificity of this branch
All patches extracted from each image are passed to classifier. 
Each patch has **4** channels (RGB + Segmentation Prediction Channel) in order to make the classifier aware of segmentation model prediction during classification of patches. During training a patch is labelled as positive if the corresponding patch in ground truth segmentation contains at least 200 pixels set to 1.

**Best configuration achieved with**:
- Focal Loss. Alpha = 0.1, Gamma = 2.0
- Patch size = 128
- Res2Net(pretrained) as classifier

