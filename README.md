# Info
In this branch a sequence of segmentation model and cyst classifier is implemented in order to train the two models together. Based on multi-scale-training branch at f3a06f4 commit.
Patch extraction is perfomed using unfold method in order to gain performances and keep gradient calculations correct, without interruptions.
Caranet (no multi-scale) in sequence with classifier. 
Images are resized by a 0.75 factor. 

Patches have 4 channels here!

**Specifically, in this branch only a fixed number of negative patches is passed to classifiier during training in order to obtain a more balanced version of the dataset. During validation and test all patches from each image are passed to be classified.