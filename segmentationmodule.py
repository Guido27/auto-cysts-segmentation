from pathlib import Path
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from utils import (
    object_from_dict,
    find_average,
    binary_mean_iou,
    extract_wrong_cysts,
    extract_real_cysts,
    refine_predicted_masks,
    save_predictions,
    extract_segmented_cysts_test_time,
    extract_patches_train_val
)

from PIL import Image
import pytorch_lightning as pl
import torchmetrics as tm
import wandb

import numpy as np
import pandas as pd
from time import time

import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models import res2net50


class SegmentCyst(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = self.hparams.model.get("name", "").lower()
        self.model = object_from_dict(hparams["model"])

        self.classifier = res2net50(pretrained=True)
        self.classifier.fc = torch.nn.Linear(
            2048, 2
        )  # changing the number of output features to 2

        self.train_images = (
            Path(self.hparams.checkpoint_callback["dirpath"])
            / "images/train_predictions"
        )
        self.val_images = (
            Path(self.hparams.checkpoint_callback["dirpath"]) / "images/val_predictions"
        )

        if not self.hparams.discard_res:
            self.train_images.mkdir(exist_ok=True, parents=True)
            self.val_images.mkdir(exist_ok=True, parents=True)

        self.loss = object_from_dict(hparams["loss"])
        self.loss_classifier = torch.nn.CrossEntropyLoss()

        self.max_val_iou = 0
        self.timing_result = pd.DataFrame(columns=["name", "time"])
        self.train_metrics = torch.nn.ModuleDict(
            {
                "iou": tm.JaccardIndex(task="binary"),
                # 'dice': tm.F1Score(task='binary'),
                "pdice": tm.F1Score(task="binary", average="samples"),
            }
        )
        self.val_metrics = torch.nn.ModuleDict(
            {
                "iou": tm.JaccardIndex(task="binary"),
                # 'dice': tm.F1Score(task='binary'),
                "pdice": tm.F1Score(task="binary", average="samples"),
            }
        )
        self.test_metrics = torch.nn.ModuleDict(
            {
                "iou": tm.JaccardIndex(task="binary"),
            }
        )
        self.epoch_start_time = []

        self.refined_results_folder = ""
        self.refined_results_folder_test = ""
        self.p_size = 64 # size of patches for classifier
        # set automatic optimization as False
        self.automatic_optimization = False

    def forward(self, batch: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        # transform to a float tensor because there are no augmentations that do it implicitly here
        batch = batch.float()

        if masks is not None:
            return self.model(batch, masks)
        else:
            return self.model(batch)

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams.optimizer,
            params=[x for x in self.model.parameters() if x.requires_grad],
        )
        opt = [optimizer]

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler["type"] == "torch.optim.lr_scheduler.LambdaLR":
                decay_epoch = 1  # how many epochs have to pass before changing the LR
                lambda1 = lambda epoch: 0.1 ** (epoch // decay_epoch)
                scheduler = object_from_dict(
                    self.hparams.scheduler, optimizer=optimizer, lr_lambda=lambda1
                )

            else:
                scheduler = object_from_dict(
                    self.hparams.scheduler, optimizer=optimizer
                )

                if type(scheduler) == ReduceLROnPlateau:
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": scheduler,
                        "monitor": "val_iou",
                    }

            return opt, [scheduler]

        return opt

    def log_images(self, features, masks, logits_, batch_idx, rate):
        # logits_ is the output of the last layer of the model
        for img_idx, (image, y_true, y_pred) in enumerate(
            zip(features, masks, logits_)
        ):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

            # image is a float tensor
            ax1.set_title("IMAGE")
            # ax1.axis('off')
            ax1.imshow((image).cpu().permute(1, 2, 0).numpy().astype(np.uint8))

            ax2.set_title("GROUND TRUTH")
            # ax2.axis('off')
            ax2.imshow(
                (y_true).permute(1, 2, 0).squeeze().cpu().numpy().astype(np.uint8),
                cmap="gray",
            )

            ax3.set_title("MODEL PREDICTION")
            # ax3.axis('off')
            y_pred = (
                (y_pred > 0.5)
                .permute(1, 2, 0)
                .squeeze()
                .cpu()
                .detach()
                .numpy()
                .astype(np.uint8)
            )
            ax3.imshow((y_pred), cmap="gray")

            # create folder if not exists
            Path("check_training").mkdir(parents=True, exist_ok=True)
            # save figure
            fig.savefig(
                f"check_training/epoch_{self.current_epoch}_batch_{batch_idx}_img_{img_idx}_rate_{rate}.png"
            )

    def save_predictions(self, predictions, images_name):
        """Save predictions of model in a batch. Use this function in training a validation.

        Parameters
        ----------
        predictions: segmentation mask (more specifically: logits) predicted from model on current image, batch of predictions
        images_name: name of predicted images in current batch
        destination_folder: where to save image, correspond to current epoch dataset folder
        """
        for pred, image_name in zip(predictions, images_name):
            pred = (
                (pred > self.hparams.test_parameters["threshold"])
                .permute(1, 2, 0)
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            Image.fromarray(pred * 255).save(
                Path(self.epoch_dataset_folder) / f"{image_name}.png"
            )

    def on_train_epoch_start(self):
        # create dataset folder for current epoch
        self.refined_results_folder = f"refined_images/train_epoch_{self.trainer.current_epoch}"
        Path(self.refined_results_folder).mkdir(parents=True, exist_ok=True)
        self.epoch_start_time.append(time())

    def get_lr(self):
        optimizer = self.optimizers()
        for param_group in optimizer.param_groups:
            return param_group["lr"]
        
    def on_train_epoch_end(self):
        self.log("epoch", float(self.trainer.current_epoch))
        #  scheduler.step() after each train epoch
        sch = self.lr_schedulers()
        sch.step()
        self.log("LR", self.get_lr(), on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.refined_results_folder_test = f"refined_images/test_results"
        Path(self.refined_results_folder_test).mkdir(parents=True, exist_ok=True)

    def training_step(self, batch, batch_idx):
        imgs_name = batch["image_id"]
        features = batch["features"] #rgb images
        masks = batch["masks"] #gt masks

        # manual steps in order to perform multi-scale training
        size_rates =[0.75] #[0.75, 1.25, 1]
        for rate in size_rates:
    
            optimizer = self.optimizers()
            optimizer.zero_grad()
      
            # ---- data prepare ----
            images = features.float()
            gts = masks
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            images_dim = 1024  # original dimension of images 1024x1024
            trainsize = int(round(images_dim * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(
                    images,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )
                gts = F.upsample(
                    gts,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )

            if self.model_name == "caranet":
                (
                    lateral_map_5,
                    lateral_map_3,
                    lateral_map_2,
                    lateral_map_1,
                ) = self.forward(images)

                # compute segmentation loss
                loss5 = self.loss(lateral_map_5, gts)
                loss3 = self.loss(lateral_map_3, gts)
                loss2 = self.loss(lateral_map_2, gts)
                loss1 = self.loss(lateral_map_1, gts)

                segmentation_loss = loss5 + loss3 + loss2 + loss1
                logits = lateral_map_5

            else:
                logits = self.forward(images)
                segmentation_loss = self.loss(logits, gts)
            # logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")

            batch_output = torch.empty(gts.shape).cuda() # refined masks computed in current batch will be saved here to compute metrics
            output_idx = 0

            # use cyst classifier on each batch image
            patches = torch.empty((1,3,self.p_size,self.p_size), dtype=torch.float32).cuda()
            labels = torch.empty((1)).type(torch.LongTensor).cuda() # LongTensor required dtype for CrossEntropyLoss
            labels.requires_grad = False
            coordinates = []
            patch_each_image = []

            # TODO perform for loops inside functions and not here in training step
            for m, p, i, n in zip(gts, logits, features, imgs_name):
                # m GT mask, i image, p segmentation predictions from model, n name of current image
                
                negative_patches_tensor, wrong_coordinates = extract_wrong_cysts(
                    m.detach().squeeze().cpu().numpy().astype(np.uint8),
                    (p > 0.5).detach().squeeze().cpu().numpy().astype(np.uint8),
                    i.detach().permute(1, 2, 0).cpu().numpy(),
                    p_size=self.p_size
                )
                # image i dimensions are permuted because has (C*H*W) shape, 
                # while slicing in extract_wrong_predictions expects to receive a H,W,C image

                positive_patches_tensor, detected_coordinates = extract_real_cysts(
                    m.detach().squeeze().cpu().numpy().astype(np.uint8),
                    (p > 0.5).detach().squeeze().cpu().numpy().astype(np.uint8),
                    i.detach().permute(1, 2, 0).cpu().numpy(),
                    p_size=self.p_size
                )

                # create labels for positive and negative patches
                positive_labels = torch.ones(positive_patches_tensor.shape[0]).type(torch.LongTensor).cuda()
                negative_labels = torch.zeros(negative_patches_tensor.shape[0]).type(torch.LongTensor).cuda()

                # concatenate patches and labels in single tensors
                patches = torch.cat((patches, negative_patches_tensor.cuda(), positive_patches_tensor.cuda()))
                labels = torch.cat((labels, negative_labels, positive_labels))
                # concatenate coordinates in the same order
                coordinates = coordinates +  wrong_coordinates + detected_coordinates 
                # save the total numnber of patches obtained from current segm. model prediction in appropriate list
                tot_patches = len(detected_coordinates) + len(wrong_coordinates)
                patch_each_image.append(tot_patches)

            #start debug
            #test new function
            patches2, labels2, coordinates2, patch_each_image2 = extract_patches_train_val(gts, logits,features)
            
            print(torch.eq(patches[1:], patches2).all())
            print(torch.eq(labels2,labels[1:]).all())
            print(coordinates == coordinates2)
            print(patch_each_image == patch_each_image2)
            #end debug
            
            # compute classifier predictions/logits
            classifier_predictions = self.classifier(patches[1:,:,:,:]) # pass patches excluding the first empty one, classifier_predictions has shape (N,2), contains logits/probabilities for each class
            
            # get predicted labels from classifier logits
            predicted_labels = torch.max(classifier_predictions, 1)[1]  # compute from raw score (logits) predictions for all patches expressed as class labels (0 or 1)
          
            # compute classifier loss over all patches in current batch
            classifier_loss = self.loss_classifier(classifier_predictions, labels[1:]) #exclude first empty label

            # training loss
            loss = segmentation_loss + classifier_loss # both computed over batch images and patches

            # refine predictions
            refined_predictions = refine_predicted_masks(logits, coordinates, patch_each_image, predicted_labels)

            
            self.log_dict(
                {
                    "segmentation_loss": segmentation_loss,
                    "classifier_loss": classifier_loss,
                    "train_loss": loss,
                }, 
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

            # passare la refined mask (logits) oppure (refined_mask > 0.5)? -> passare logits, il thresholding lo fa gia la metric da sola
            for metric_name, metric in self.train_metrics.items():
                metric(refined_predictions, gts.int())
                self.log(
                    f"train_{metric_name}",
                    metric,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )

            self.manual_backward(loss)

            self.clip_gradients(
                optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )

            optimizer.step()


            # scheduler step after each optimizer.step(), i.e. one for each batch in each resize
            # sch = self.lr_schedulers()
            # sch.step()

        # scheduler step after entire batch (after for loop on rates for each batch)
        # sch = self.lr_schedulers()
        # sch.step()

        return {
                "segmentation_loss": segmentation_loss,
                "classifier_loss": classifier_loss,
                "train_loss": loss
                }

    #TODO aggiornare val_step come train step se funziona tutto
    def validation_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]
        imgs_name = batch["image_id"]

        if self.model_name in ["uacanet", "pranet"]:
            logits = self.forward(features, masks)
            loss = logits["loss"]
            logits = logits["pred"]

        # CaraNet
        elif self.model_name in ["caranet"]:
            lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1 = self.forward(
                features
            )

            loss5 = self.loss(lateral_map_5, masks)
            loss3 = self.loss(lateral_map_3, masks)
            loss2 = self.loss(lateral_map_2, masks)
            loss1 = self.loss(lateral_map_1, masks)

            segmentation_loss = loss5 + loss3 + loss2 + loss1
            logits = lateral_map_5

        else:
            logits = self.forward(features)
            segmentation_loss = self.loss(logits, masks)

        #logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")

        batch_output = torch.empty(masks.shape).cuda()
        output_idx = 0

        # extract segmented areas and run classifier on them
        for p, i, m in zip(logits, features, masks):
            
            # extract wrng predictions
            negative_patches_tensor, wrong_coordinates = extract_wrong_cysts(
                    m.detach().squeeze().cpu().numpy().astype(np.uint8),
                    (p > 0.5).detach().squeeze().cpu().numpy().astype(np.uint8),
                    i.detach().permute(1, 2, 0).cpu().numpy()
                )

            #extract positive patches
            positive_patches_tensor, detected_coordinates = extract_real_cysts(
                m.detach().squeeze().cpu().numpy().astype(np.uint8),
                (p > 0.5).detach().squeeze().cpu().numpy().astype(np.uint8),
                i.detach().permute(1, 2, 0).cpu().numpy(),
            )

            # if no segmented areas in segmentation model predicion -> avoid classifier in order to don't create nan classifier loss which breaks training
            if not (len(detected_coordinates) == 0 and len(wrong_coordinates) == 0):

                # create labels for positive and negative patches
                positive_labels = torch.ones(positive_patches_tensor.shape[0])
                negative_labels = torch.zeros(negative_patches_tensor.shape[0])

                # concatenate patches and labels in single tensors
                patches = torch.cat(
                    (positive_patches_tensor, negative_patches_tensor), dim=0
                ).cuda()
                labels = (
                    torch.cat((positive_labels, negative_labels), dim=0)
                    .type(
                        torch.LongTensor
                    )  # LongTensor is required for BCE loss with labels
                    .cuda()
                )

                # compute predictions over patches from classifier
                classifier_predictions = torch.empty((1, 2)).cuda()
                for patch in patches:
                    r = self.classifier(
                        patch.unsqueeze(0)
                    )  # r shape is (1, 2), 2 classes
                    classifier_predictions = torch.cat((classifier_predictions, r))
                    # classifier_predictions shape is (N,2) where N is the number of predictions and 2 is the number of classes,
                    # each column represent a class and the value inside is the score (probability) computed for that specific class,
                    # contains logits basically

                # refine segmentation mask: remove segmented areas classified as False/Not-Cyst from classifier in segmentation mask
                predicted_classes = torch.max(classifier_predictions[1:], 1)[1]  # compute from raw score (logits) predictions for all patches expressed as class labels (0 or 1)
                coordinates = torch.tensor(detected_coordinates + wrong_coordinates).cuda()

                to_erase_predictions = coordinates[ predicted_classes == 0]  # use predictions on patches as mask label to get coordinates of ones classified as False/0
                refined_mask = refine_mask(p, to_erase_predictions)

                batch_output[output_idx] = refined_mask
                output_idx = output_idx + 1

                classifier_loss = self.loss_classifier(
                    classifier_predictions[1:], labels
                )

                # compute general loss
                loss = segmentation_loss + classifier_loss
            else:
                # TODO is it correct? 
                loss = segmentation_loss
                classifier_loss = 0

        # don't save predictions in val set for the moment 

        self.log_dict({"val_segmentation_loss": segmentation_loss,
                    "val_classifier_loss": classifier_loss,
                    "val_loss": loss}, 
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,)
       
        for metric_name, metric in self.val_metrics.items():
            metric(batch_output, masks.int()) # compute metrics on refined masks
            self.log(f"val_{metric_name}", metric, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]

        t0 = time()
        if self.model_name in ["uacanet", "pranet"]:
            logits = self.forward(features, masks)
            logits = logits["pred"]
        elif self.model_name in ["caranet"]:
            logits, lateral_map_3, lateral_map_2, lateral_map_1 = self.forward(features)
        else:
            logits = self.forward(features)

        timing = [time() - t0, features.shape[0]]
        
        batch_output = torch.empty(masks.shape).cuda()
        output_idx = 0
        # extract segmented areas and run classifier on them
        for p, i, m in zip(logits, features, masks):
            patches, coordinates = extract_segmented_cysts_test_time(
                (p>0.5).detach().squeeze().cpu().numpy().astype(np.uint8),
                i.detach().permute(1, 2, 0).cpu().numpy(),)
           
            patches = patches.cuda() #move patches to gpu

            # compute predictions over patches from classifier
            classifier_predictions = torch.empty((1, 2)).cuda()
            for patch in patches:
                r = self.classifier(
                    patch.unsqueeze(0)
                )  # r shape is (1, 2), 2 classes
                classifier_predictions = torch.cat((classifier_predictions, r))
                # classifier_predictions shape is (N,2) where N is the number of predictions and 2 is the number of classes,
                # each column represent a class and the value inside is the score (probability) computed for that specific class,
                # contains logits basically
            
            # refine segmentation mask: remove segmented areas classified as False/Not-Cyst from classifier in segmentation mask
            predicted_classes = torch.max(classifier_predictions[1:], 1)[1]  # compute from raw score (logits) predictions for all patches expressed as class labels (0 or 1)
            coordinates = torch.tensor(coordinates).cuda()
            
            to_erase_predictions = coordinates[ predicted_classes == 0]  # use predictions on patches as mask label to get coordinates of ones classified as False/0
            refined_mask = refine_mask(p, to_erase_predictions)

            batch_output[output_idx] = refined_mask
            output_idx = output_idx + 1

        for i in range(features.shape[0]):
            name = batch["image_id"][i]
            p = logits[i][0]
            logits_ = batch_output[i][0] #refined masks
            mask = masks[i][0] #gt masks

            save_predictions(
                        mask.detach().squeeze().cpu().numpy().astype(np.uint8),
                        (p > 0.5).detach().squeeze().cpu().numpy().astype(np.uint8),
                        (logits_>0.5).detach().squeeze().cpu().numpy().astype(np.uint8),
                        name,
                        Path(self.refined_results_folder_test)
                    )
            
            logits_ = (
                logits_.cpu().numpy() > self.hparams.test_parameters["threshold"]
            ).astype(np.uint8)
            Image.fromarray(logits_ * 255).save(
                self.hparams.checkpoint_callback["dirpath"]
                / "result"
                / "test"
                / f"{name}.png"
            )

        self.timing_result.loc[len(self.timing_result)] = timing
        for metric_name, metric in self.test_metrics.items():
            metric(batch_output, masks.int())
            self.log(f"test_{metric_name}", metric, on_step=True, on_epoch=True)
