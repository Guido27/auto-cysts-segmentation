from pathlib import Path
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import object_from_dict, find_average, binary_mean_iou

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

class SegmentCyst(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = self.hparams.model.get('name', '').lower()
        self.model = object_from_dict(hparams["model"])
        
        self.train_images = Path(self.hparams.checkpoint_callback["dirpath"]) / "images/train_predictions"
        self.val_images =  Path(self.hparams.checkpoint_callback["dirpath"]) / "images/val_predictions"

        if not self.hparams.discard_res:
            self.train_images.mkdir(exist_ok=True, parents=True)
            self.val_images.mkdir(exist_ok=True, parents=True)
         
        self.loss = object_from_dict(hparams["loss"])
        self.max_val_iou = 0
        self.timing_result = pd.DataFrame(columns=['name', 'time'])
        self.train_metrics = torch.nn.ModuleDict({
            'iou': tm.JaccardIndex(task='binary'),
            # 'dice': tm.F1Score(task='binary'),
            'pdice': tm.F1Score(task='binary', average='samples'),
        })
        self.val_metrics = torch.nn.ModuleDict({
            'iou': tm.JaccardIndex(task='binary'),
            # 'dice': tm.F1Score(task='binary'),
            'pdice': tm.F1Score(task='binary', average='samples'),
        })
        self.test_metrics = torch.nn.ModuleDict({
            'iou': tm.JaccardIndex(task='binary'),
        })
        self.epoch_start_time = []

        # set automatic optimization as False
        self.automatic_optimization = False

    def forward(self, batch: torch.Tensor, masks: torch.Tensor=None) -> torch.Tensor:

        #transform to a float tensor because there are no augmentations that do it implicitly here
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
        optimizer = [optimizer]
        
        if self.hparams.scheduler is not None:
            scheduler = object_from_dict(self.hparams.scheduler, optimizer=optimizer)

            if type(scheduler) == ReduceLROnPlateau:
                    return {
                       'optimizer': optimizer,
                       'lr_scheduler': scheduler,
                       'monitor': 'val_iou'
                   }
            return optimizer, [scheduler]
        return optimizer
    
    def log_images(self, features, masks, logits_, batch_idx):
        # logits_ is the output of the last layer of the model
        for img_idx, (image, y_true, y_pred) in enumerate(zip(features, masks, logits_)):
            
            fig,(ax1, ax2, ax3) = plt.subplots(1,3,figsize = (10,5))

            # image is a float tensor
            ax1.set_title('IMAGE')
            #ax1.axis('off')
            ax1.imshow((image).cpu().permute(1,2,0).numpy().astype(np.uint8))

            ax2.set_title('GROUND TRUTH')
            #ax2.axis('off')
            ax2.imshow((y_true).permute(1,2,0).squeeze().cpu().numpy().astype(np.uint8),cmap = 'gray')

            ax3.set_title('MODEL PREDICTION')
            #ax3.axis('off')
            y_pred = (y_pred > 0.5).permute(1,2,0).squeeze().cpu().detach().numpy().astype(np.uint8)
            ax3.imshow((y_pred),cmap = 'gray')

            # create folder if not exists
            Path("check_training").mkdir(parents=True, exist_ok=True)
            # save figure
            fig.savefig(f'check_training/epoch_{self.current_epoch}_batch_{batch_idx}_img_{img_idx}.png')

    def on_epoch_start(self):
        self.epoch_start_time.append(time())
    
    def training_step(self, batch, batch_idx):

        features = batch["features"]
        masks = batch["masks"]
    
        # CaraNet    
        size_rates = [0.75, 1, 1.25]
        for rate in size_rates:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            
            # ---- data prepare ----
            images = features.float()
            gts = masks
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            images_dim = 1024 # original dimension of images 1024x1024
            trainsize = int(round(images_dim*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1 = self.forward(images)
            
            #compute loss
            loss5 = self.loss(lateral_map_5, gts)
            loss3 = self.loss(lateral_map_3, gts)
            loss2 = self.loss(lateral_map_2, gts)
            loss1 = self.loss(lateral_map_1, gts)
            loss = loss5 +loss3 + loss2 + loss1
            
            logits = lateral_map_5
    
            #logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")

            if batch_idx == 0 and self.trainer.current_epoch % 2 == 0:
                self.log_images(images, gts, logits, batch_idx)

            for metric_name, metric in self.train_metrics.items():
                metric(logits, gts.int())
                self.log(f"train_{metric_name}", metric, on_step=True, on_epoch=True, prog_bar=True)

            self.log("train_loss", loss)
            self.log("lr", self._get_current_lr())

            self.manual_backward(loss)

            # clip gradients
            self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

            optimizer.step()

            sch= self.lr_schedulers()
            sch.step()

            return {"loss": loss}
        

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        
        if torch.cuda.is_available(): return torch.Tensor([lr])[0].cuda()
        return torch.Tensor([lr])[0]

    def validation_step(self, batch, batch_id):
        features = batch["features"].float()
        masks = batch["masks"]
        
        if self.model_name in ['uacanet', 'pranet']:
            logits = self.forward(features, masks)
            loss = logits['loss'] 
            logits = logits['pred']

        # CaraNet    
        elif self.model_name in ['caranet']:
            
            lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1 = self.forward(features)
            
            loss5 = self.loss(lateral_map_5, masks)
            loss3 = self.loss(lateral_map_3, masks)
            loss2 = self.loss(lateral_map_2, masks)
            loss1 = self.loss(lateral_map_1, masks)

            loss = loss5 +loss3 + loss2 + loss1            
            logits = lateral_map_5

        else:
            logits = self.forward(features)
            loss = self.loss(logits, masks)
            
        logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")
        
        if not self.hparams.discard_res and wandb.run is not None:
            if self.trainer.current_epoch % 5 == 0:
                class_labels = {0: "background", 1: "cyst"}
                mask_img = wandb.Image(
                    features[0, :, :, :],
                    masks={
                        "predictions": {
                            "mask_data": logits_[0, 0, :, :],
                            "class_labels": class_labels,
                        },
                        "groud_truth": {
                            "mask_data": masks.cpu().detach().numpy()[0, 0, :, :],
                            "class_labels": class_labels,
                        },
                    },
                )
                self.logger.experiment.log({"val_images": [mask_img]}, commit=False)

        self.log("val_loss", loss)
        # self.log("val_iou", result["val_iou"])
        for metric_name, metric in self.val_metrics.items():
            metric(logits, masks.int())
            self.log(f"val_{metric_name}", metric, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        self.log("epoch", float(self.trainer.current_epoch))

    # def on_train_end(self):
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    #     self.train_metrics['iou'].plot(ax=ax[0])
    #     self.val_metrics['iou'].plot(ax=ax[0])
    #     self.val_metrics['dice'].plot(ax=ax[0])
    #     ax[0].legend()

    #     self.epoch_start_time = np.array(self.epoch_start_time) - self.epoch_start_time[0]
    #     ax[1].plot(self.epoch_start_time, label='Epoch duration')
    #     ax[1].set_xlabel('Epoch')
    #     ax[1].set_ylabel('Time [s]')
    #     ax[1].legend()

    #     fig.savefig(self.hparams.checkpoint_callback['dirpath'] / 'metrics.png')

    def test_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]
    
        t0 = time()
        if self.model_name in ['uacanet', 'pranet']:
            logits = self.forward(features, masks)
            logits = logits['pred']
        elif self.model_name in ['caranet']:
            logits, lateral_map_3, lateral_map_2, lateral_map_1  = self.forward(features)
        else:
            logits = self.forward(features)
    
        
        timing = [time()-t0, features.shape[0]]
        # result["test_iou"] = binary_mean_iou(logits, masks)
        for i in range(features.shape[0]):
            name = batch["image_id"][i]
            logits_ = logits[i][0]

            logits_ = (logits_.cpu().numpy() > self.hparams.test_parameters['threshold']).astype(np.uint8)
            Image.fromarray(logits_*255).save(self.hparams.checkpoint_callback['dirpath'] /'result'/'test'/f"{name}.png")

        self.timing_result.loc[len(self.timing_result)] = timing
        for metric_name, metric in self.test_metrics.items():
            metric(logits, masks.int())
            self.log(f"test_{metric_name}", metric, on_step=True, on_epoch=True)
    