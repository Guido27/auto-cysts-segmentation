# Info
In this branch a sequence of segmentation model and cyst classifier is implemented in order to train the two models together. Based on multi-scale-training branch at f3a06f4 commit.
Patch extraction is perfomed using unfold method in order to gain performances and keep gradient calculations correct, without interruptions.
Caranet (no multi-scale) in sequence with classifier. 
Images are resized by a 0.75 factor. 

**Specifically here only patches associated with segmentation predictions are passed to classifier not all patches from each image.**