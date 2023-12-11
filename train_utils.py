import shutil
import yaml
import json

from utils import *
from segmentationmodule import SegmentCyst
from dataloaders import CystDataModule

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb



def train(args, hparams, name=None):
    ##########################################################
    wandb.init(project="cyst_segmentation_SequenceModels",
            tags=[args.tag] if args.tag else None, reinit=True,
            name=None
            ) if args.wb else None
    # name = wandb.run.name if args.wb else "foo"
    hparams = init_training(args, hparams, name, tiling=getattr(args, 'tiling', None))

    #################
    if name == "foo":
        print(">> delete everything in the folder")
        for f in hparams["checkpoint_callback"]["dirpath"].glob("*"):
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
                
    test_preds = hparams["checkpoint_callback"]["dirpath"]/'result'/'test'
    test_preds.mkdir(exist_ok=True, parents=True)

    success = hparams["checkpoint_callback"]["dirpath"] / ".success"

    if success.exists() and any(test_preds.glob('*png')):
        return

    if args.model:
        with open("configs/models.yaml") as f:
            models = yaml.load(f, Loader=yaml.SafeLoader)
            if args.model in models.keys():
                hparams["model"] = models[args.model]["model"]
            else:
                raise NotImplemented("Model not implemented")
        delattr(args, 'model')

    if args.loss:
        with open("configs/losses.yaml") as f:
            losses = yaml.load(f, Loader=yaml.SafeLoader)
            if args.loss in losses.keys():
                hparams["loss"] = losses[args.loss]
            else:
                raise NotImplemented("Loss function not implemented")
        delattr(args, 'loss')

    earlystopping_callback = EarlyStopping(**hparams["earlystopping_callback"])
    checkpoint_callback = ModelCheckpoint(**hparams["checkpoint_callback"])

    max_epochs = hparams['train_parameters']['epochs']

    data = CystDataModule(**dict(hparams, **args.__dict__))
    model = SegmentCyst(**hparams,
                    discard_res=~args.save_results,
                    )
    
    with (hparams["checkpoint_callback"]["dirpath"] / "split_samples.json").open('w') as file:
        json.dump({
            'train': str_from_samples(data.train_samples),
            'val': str_from_samples(data.val_samples),
            'test': str_from_samples(data.test_samples)
        }, file)
    print(f'\nSaving in {hparams["checkpoint_callback"]["dirpath"]}\n')
    
    logger = WandbLogger() if args.wb else None
    if logger:
        logger.log_hyperparams(hparams)
        # logger.watch(model, log='all', log_freq=1)

    if getattr(args, 'seed', None) is not None:
        pl.seed_everything(args.seed)
        
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 0,
        devices=[0],#torch.cuda.device_count(),
        accumulate_grad_batches=args.acc_grad if hasattr(args, 'acc_grad') else 1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback,
                earlystopping_callback,
                ],
        # strategy='dp',
        #gradient_clip_val=5.0,
        num_sanity_val_steps=0, #set to 0 in order to to avoid validation steps before training starts
        sync_batchnorm=True,
        logger=logger,
        precision=16,
    )
            
    if not (success).exists():
        trainer.fit(model, datamodule=data)
        if logger:
            model.logger.experiment.log({"max_val_iou": model.max_val_iou})

        with open(success, "w") as f:
            f.write("")


    print("\n\nTraining completed\n\n")
    
    if not any(test_preds.glob('*png')):
        # getting back the best model
        model = SegmentCyst.load_from_checkpoint(next(hparams["checkpoint_callback"]["dirpath"].glob("*.ckpt")))
        trainer.test(model, datamodule=data)
