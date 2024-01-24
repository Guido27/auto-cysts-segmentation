from pathlib import Path
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from utils import (
    object_from_dict,
    find_average,
    binary_mean_iou,
    refine_predicted_masks,
    save_images,
    extract_patches,
    unfold_patches,
    refine_predictions_unfolding
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
from losses import FocalLoss
from efficientnet_pytorch import EfficientNet

class SegmentCyst(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = self.hparams.model.get("name", "").lower()
        self.model = object_from_dict(hparams["model"])

        # classifier settings
        if self.hparams.classifier == "RES2NET":
            self.classifier = res2net50(pretrained=True)
            self.classifier.fc = torch.nn.Linear(2048, 2)  # changing the number of output classes to 2

        if self.hparams.classifier == "EFFICIENTNET":
            self.classifier = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)

        if self.hparams.c_loss == "CE":
            self.weight = torch.tensor([0.1, 1.50]) # class 0, class 1 
            self.loss_classifier = torch.nn.CrossEntropyLoss(weight=self.weight)

        if self.hparams.c_loss == "Focal":
            self.loss_classifier = FocalLoss(gamma = self.hparams.gamma, alpha=self.hparams.alpha)
        # end classifier setings 
            
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
        self.patch_size = self.hparams.patch_size # size of patches for classifier
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
        #choosing a optimizer for classifier
        c_optimizer = torch.optim.Adam(self.parameters() , lr = 1e-4)
        #choosing LR scheduler for classifier
        c_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer = c_optimizer, T_0 = 10, T_mult = 2)

        # choosing a optimizer for sementation model
        s_optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)
        decay_epoch = 1  # how many epochs have to pass before changing the LR
        lambda1 = lambda epoch: 0.1 ** (epoch // decay_epoch)
        s_sch = torch.optim.lr_scheduler.LambdaLR(optimizer=s_optimizer, lr_lambda=lambda1)

        return ({"optimizer": s_optimizer, "lr_scheduler": s_sch}, {"optimizer": c_optimizer, "lr_scheduler": c_sch})

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

    def get_segmentation_lr(self):
        s_optimizer, c_optimizer = self.optimizers() # optimizer = self.optimizers()
        for param_group in s_optimizer.param_groups: # for param_group in optimizer.param_group
            return param_group["lr"]
        
    def on_train_epoch_end(self):
        self.log("epoch", float(self.trainer.current_epoch))
        #  scheduler.step() after each train epoch
        #sch = self.lr_schedulers()
        #sch.step()
        segmentation_scheduler, _ = self.lr_schedulers()
        segmentation_scheduler.step()
        self.log("segmentation_lr", self.get_segmentation_lr(), on_step=False, on_epoch=True, prog_bar=True)

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
    
            #optimizer = self.optimizers()
            #optimizer.zero_grad()
            s_optimizer, c_optimizer = self.optimizers()
            s_optimizer.zero_grad()
            c_optimizer.zero_grad()
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

            patches, labels = unfold_patches(gts,images, size = self.patch_size, stride = self.patch_size)
    
            # compute classifier predictions/logits
            classifier_predictions = self.classifier(patches) # pass patches excluding the first empty one, classifier_predictions has shape (N,2), contains logits/probabilities for each class
            
            # get predicted labels from classifier logits
            predicted_labels = torch.max(classifier_predictions, 1)[1]  # compute from raw score (logits) predictions for all patches expressed as class labels (0 or 1)
            
            # compute classifier loss over all patches in current batch
            if self.hparams.c_loss == "CE":
                classifier_loss = self.loss_classifier(classifier_predictions, labels)
            if self.hparams.c_loss == "Focal":
               classifier_loss = self.loss_classifier(classifier_predictions, labels)

            # training loss
            loss = segmentation_loss + classifier_loss # both computed over batch images and patches

            # refine predictions
            refined_predictions = refine_predictions_unfolding(logits, predicted_labels, size = self.patch_size, stride = self.patch_size)
            #refined_predictions = refine_predicted_masks(logits, coordinates, patch_each_image, predicted_labels)

            if self.hparams.debug:
                print("Predicted labels:")
                print(predicted_labels)
                print("True Labels:")
                print(labels)
            
            self.log_dict(
                {
                    "segmentation_loss": segmentation_loss,
                    "classifier_loss": classifier_loss,
                    "train_loss": loss,
                }, 
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            # passare la refined mask (logits) oppure (refined_mask > 0.5)? -> passare le logits, il thresholding lo fa gia la metric da sola
            for metric_name, metric in self.train_metrics.items():
                metric(refined_predictions, gts.int())
                self.log(
                    f"train_{metric_name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

            self.manual_backward(loss)

            #self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            self.clip_gradients(s_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            self.clip_gradients(c_optimizer,gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            
            c_optimizer.step()
            s_optimizer.step()

            # scheduler step after each optimizer.step(), i.e. one for each batch in each resize
            # sch = self.lr_schedulers()
            # sch.step()

        # scheduler step after entire batch (after for loop on rates for each batch)
        # sch = self.lr_schedulers()
        # sch.step()
        s_sch, c_sch = self.lr_schedulers()
        c_sch.step()

        # save first image of current batch every 5 batch, not all batch because it would saturate colab memory
        if batch_idx % 5 == 0:
                save_images(masks[:1],logits[:1], refined_predictions[:1],f"batch_idx_{batch_idx:03}",Path(self.refined_results_folder))


        return {
                "segmentation_loss": segmentation_loss,
                "classifier_loss": classifier_loss,
                "train_loss": loss
                }

    def validation_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]
        imgs_name = batch["image_id"]

        # manual steps in order to perform multi-scale training
        size_rates =[0.75] #[0.75, 1.25, 1]
        for rate in size_rates:
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

            patches, labels = unfold_patches(gts,images, size=self.patch_size, stride=self.patch_size) # here labels are used only to compute validation loss
    
            # compute classifier predictions/logits
            classifier_predictions = self.classifier(patches) # classifier_predictions has shape (N,2), contains logits/probabilities for each class
            
            # get predicted labels from classifier logits
            predicted_labels = torch.max(classifier_predictions, 1)[1]  # compute from raw score (logits) predictions for all patches expressed as class labels (0 or 1)
            
            # compute classifier loss over all patches in current batch
            if self.hparams.c_loss == "CE":
                classifier_loss = self.loss_classifier(classifier_predictions, labels)
            if self.hparams.c_loss == "Focal":
                classifier_loss = self.loss_classifier(classifier_predictions, labels)

            # training loss
            loss = segmentation_loss + classifier_loss # both computed over batch images and patches

            # refine predictions
            refined_predictions = refine_predictions_unfolding(logits, predicted_labels, size=self.patch_size, stride=self.patch_size)

        self.log_dict({"val_segmentation_loss": segmentation_loss,
                    "val_classifier_loss": classifier_loss,
                    "val_loss": loss}, 
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,)
       
        for metric_name, metric in self.val_metrics.items():
            metric(refined_predictions, gts.int()) # compute metrics on refined masks
            self.log(f"val_{metric_name}", metric, on_step=False, on_epoch=True)
        
        save_images(gts[:1],logits[:1], refined_predictions[:1],f"val_batch_idx_{batch_id:03}",Path(self.refined_results_folder))


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

        size_rates =[0.75] #[0.75, 1.25, 1]
        for rate in size_rates:
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

                logits = lateral_map_5

            else:
                logits = self.forward(images)

            patches = unfold_patches(gts,images, test=True, size=self.patch_size, stride = self.patch_size) # here labels are used only to compute validation loss
    
            # compute classifier predictions/logits
            classifier_predictions = self.classifier(patches) # classifier_predictions has shape (N,2), contains logits/probabilities for each class
            
            # get predicted labels from classifier logits
            predicted_labels = torch.max(classifier_predictions, 1)[1]  # compute from raw score (logits) predictions for all patches expressed as class labels (0 or 1)

            # refine predictions
            refined_predictions = refine_predictions_unfolding(logits, predicted_labels, size=self.patch_size, stride=self.patch_size)

            save_images(gts[:1],logits[:1], refined_predictions[:1],f"test_idx_{batch_id:03}",Path(self.refined_results_folder_test))

        for i in range(features.shape[0]):
            name = batch["image_id"][i]
            p = logits[i][0]
            logits_ = refined_predictions[i][0] #refined masks
            mask = masks[i][0] #gt masks

           
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
            metric(refined_predictions, gts.int())
            self.log(f"test_{metric_name}", metric, on_step=True, on_epoch=True)
