import argparse
import torch
from pathlib import Path

import pydoc
from typing import Any, Union, Dict, List, Tuple,Optional
from zipfile import ZipFile
from easydict import EasyDict as ed
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GroupKFold
import cv2
import re
from scipy.sparse import csr_matrix
from pytorch_toolbelt.utils.torch_utils import image_to_tensor
import matplotlib.pyplot as plt


import numpy as np
import torch
import wandb


def get_id2_file_paths(path: Union[str, Path]) -> Dict[str, Path]:
    return {x.stem: x for x in Path(path).glob("*.*")}



def date_to_exp(date):
    date_exps = {'0919': 1, '1019': 2, '072020':3, '092020':3, '122020':4, '0721':5}
    date = ''.join((date).split('.')[1:])
    return date_exps[date]



all_treats = {'ctrl', 't3', 'triac', 't4', 'tetrac', 'resv', 'dbd', 'lm609', 'uo', 'dbd+t4', 'uo+t4', 'lm609+t4', 'lm609+10ug.ml', 'lm609+2.5ug.ml'}

def simplify_names(filename):
    unpack = re.split(' {1,}_?|_', filename.strip())
    
    date_idx = [i for i, item in enumerate(unpack) if re.search('[0-9]{1,2}.[0-9]{1,2}.[0-9]{2,4}', item)][0]
    unpack = unpack[date_idx:]
    date = unpack[0]
    treatment = [x.upper() for x in unpack if x.lower() in all_treats][-1]

    side = [s for s in unpack if re.match('A|B', s)]
    side = side[0] if side else 'U'

    zstack = [s.lower() for s in unpack if re.match('[24]0[xX][0-9]{1,2}', s)][0]
    alt_zstack = [s for s in unpack if re.match('\([0-9]{1,}\)', s)]
    if alt_zstack: zstack = zstack.split('x')[0] + 'x' + alt_zstack[0][1:-1]
    z1, z2 = zstack.split('x')
    zstack = f"{z1}x{int(z2):02}"

    tube = [n for n in unpack if re.fullmatch('[0-9]*', n)][0]
    
    return date, treatment, tube, zstack, side


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_samples(image_path: Path, mask_path: Path) -> List[Tuple[Path, Path]]:
    """Couple masks and images.

    Args:
        image_path:
        mask_path:

    Returns:
    """

    image2path = get_id2_file_paths(image_path)
    mask2path = get_id2_file_paths(mask_path)

    return [(image_file_path, mask2path[file_id]) for file_id, image_file_path in image2path.items()]


def str_from_samples(samples, onlyname=False):
    if not samples: return []
    extr = lambda st: st.name if onlyname else str(st)
    return [(extr(p[0]), extr(p[1])) for p in samples]


def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()


def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def state_dict_from_disk(
    file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034
    return pydoc.locate(object_type)(**kwargs) if pydoc.locate(object_type) is not None else pydoc.locate(object_type.rsplit('.', 1)[0])(**kwargs)


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def load_rgb(image_path: Union[Path, str]) -> np.array:
    """Load RGB image from path.
    Args:
        image_path: path to image
        lib: library used to read an image.
            currently supported `cv2` and `jpeg4py`
    Returns: 3 channel array with RGB image
    """
    if Path(image_path).is_file():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    raise FileNotFoundError(f"File not found {image_path}")


def load_mask(path):
    im = str(path)
    return (cv2.imread(im, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

def load_mask_resized(path, size=768):
    im = str(path)
    mask = (cv2.imread(im, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    return cv2.resize(mask, (size,size), interpolation = cv2.INTER_AREA)


def binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor, EPSILON = 1e-15) -> torch.Tensor:
    output = (logits > 0.5).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()
    union = targets.sum() + output.sum() - intersection
    result = (intersection + EPSILON) / (union + EPSILON)

    return result


#######################
####                  #
#### Training utils   #
####                  #
#######################

def init_training(args, hparams, name, tiling=False):
    """
    Variable "args" is supposed to be the argparse output, while the hparams is the dictionary
    from the configuration file, which will be eventually updated
    """
    
    if type(args) == dict:
        args = ed(args)

    if args.dataset != 'nw' and args.wb and not tiling:
        dataset = wandb.run.use_artifact(f'smonaco/rene-policistico-artifacts/dataset:{args.dataset}', type='dataset')
        data_dir = dataset.download()

        if not (Path(data_dir) / "images").exists():
            zippath = next(Path(data_dir).iterdir())
            with ZipFile(zippath, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

        hparams["image_path"] = Path(data_dir) / "images"
        hparams["mask_path"] = Path(data_dir) / "masks"
    elif tiling:
        data_dir = "artifacts/tiled-dataset:v0"
        hparams["image_path"] = Path(data_dir) / "images"
        hparams["mask_path"] = Path(data_dir) / "masks"
    else:
        hparams["image_path"] = Path(hparams["image_path"])
        hparams["mask_path"] = Path(hparams["mask_path"])
    
    if hasattr(args, 'use_scheduler') and not args.use_scheduler:
        hparams['scheduler'] = None

    hparams["checkpoint_callback"]["dirpath"] = Path(hparams["checkpoint_callback"]["dirpath"])
    hparams["checkpoint_callback"]["dirpath"] /= name
    hparams["checkpoint_callback"]["dirpath"].mkdir(exist_ok=True, parents=True)

    if hasattr(args, 'noG_preprocessing'):
        hparams["noG_preprocessing"] = args.noG_preprocessing
    
    hparams['seed'] = args.seed
    hparams['tube'] = args.tube

    hparams['debug'] = args.debug
    hparams['patch_size'] = args.patch_size
    hparams['classifier'] = args.classifier
    hparams['c_loss'] = args.c_loss
    hparams['gamma'] = args.gamma
    hparams['alpha'] = args.alpha
    

    print("---------------------------------------")
    print("        Running Crossvalidation        ")
    if args.tiling:
        print("         with tiled dataset        ")
    if args.model is not None:
        print(f"         model: {args.model}  ")
    if args.exp is not None:
        print(f"           exp: {args.exp}  ")
    if args.tube is not None:
        print(f"         tube: {args.tube}  ")
    print(f"         seed: {args.seed}           ")
    print(f"         fold: {args.k}       ")
    print("---------------------------------------\n")
    
    return hparams



def split_dataset(hparams):
    samples = get_samples(hparams["image_path"], hparams["mask_path"])
    k=getattr(hparams, 'k', 0)
    test_exp=getattr(hparams, 'exp', None)
    leave_one_out=getattr(hparams, 'tube', None)
    strat_nogroups=getattr(hparams, 'stratify_fold', None)
    single_exp=getattr(hparams, 'single_exp', None)
    
    ##########################################################
    if single_exp == 1:
        samples = [u for u in samples if "09.19" in u[0].stem]
    if single_exp == 2:
        samples = [u for u in samples if "10.19" in u[0].stem]
    if single_exp == 3:
        samples = [u for u in samples if "07.2020" in u[0].stem or "09.2020" in u[0].stem]
    if single_exp == 4:
        samples = [u for u in samples if "12.2020" in u[0].stem]
#         samples = [u for u in samples if "ctrl 11" in u[0].stem.lower() or "t4" in u[0].stem.lower()]
    if single_exp == 5:
        samples = [u for u in samples if "07.21" in u[0].stem]
    ##########################################################
    
    names = [file[0].stem for file in samples]

    unpack = [simplify_names(name) for name in names]
    df = pd.DataFrame({
        "filename": names,
        "treatment": [u[1] for u in unpack],
        "exp": [date_to_exp(u[0]) for u in unpack],
        "tube": [u[2] for u in unpack],
    })
#     df["te"] = df.treatment + '_' + df.exp.astype(str)
    df["te"] = (df.treatment + '_' + df.exp.astype(str) + '_' + df.tube.astype(str)).astype('category')
    
    if test_exp is not None or leave_one_out is not None:
        if leave_one_out is not None:
            tubes = df[['exp','tube']].astype(int).sort_values(by=['exp', 'tube']).drop_duplicates().reset_index(drop=True).xs(leave_one_out)
            test_idx = df[(df.exp == tubes.exp)&(df.tube == str(tubes.tube))].index     
        else:
            test_idx = df[df.exp == test_exp].index
    
        test_samp = [x for i, x in enumerate(samples) if i in test_idx]
        samples = [x for i, x in enumerate(samples) if i not in test_idx]
        df = df.drop(test_idx)
    else:
        test_samp = None
        
    if strat_nogroups:
        skf = StratifiedKFold(n_splits=5, random_state=hparams["seed"], shuffle=True)
        train_idx, val_idx = list(skf.split(df.filename, df.te))[k]
    else:
        df, samples = shuffle(df, samples, random_state=hparams["seed"])
        gkf = GroupKFold(n_splits=5)
        train_idx, val_idx = list(gkf.split(df.filename, groups=df.te))[k]
    
    train_samp = [tuple(x) for x in np.array(samples)[train_idx]]
    val_samp = [tuple(x) for x in np.array(samples)[val_idx]]
    
    return {
        "train": train_samp,
        "valid": val_samp,
        "test": test_samp
    }

### Functions useful for classifier after segmentation model

def save_images(gt_masks, segmented_masks, refined_masks, image_name, path):
    """Save images of predicted segmentation mask, gt mask and refined mask of entire batch.
    Parameters
    ----------
    gt_masks: tensor of shape (B,1,S,S), where S is image and mask size, B batch size. Contains the ground truth masks associated with current batch images. Expect a mask of float values, no logits! 
    segmented_masks: segmented prediction logits, tensor with same shape as gt_masks. Have to be logits!
    refined_masks: refined segmented prediction logits, tensor with same shape as gt_masks Have to be logits!
    image_name: name of the image file generated
    path: Path object containing the path to save image
    """
    number_of_images = segmented_masks.shape[0]
    
    if(number_of_images > 1):
        f, cols = plt.subplots(number_of_images, 3, figsize=(15,20))
        cols[0,0].set_title('GT masks')
        cols[0,1].set_title("Predicted masks")
        cols[0,2].set_title("Refined masks")
        for (ax1, ax2, ax3), m, p, r in  zip(cols,gt_masks,segmented_masks,refined_masks):
            ax1.imshow(m.detach().squeeze().cpu().numpy().astype(np.uint8)*255, cmap='gray') 
            ax2.imshow((p>.5).detach().squeeze().cpu().numpy().astype(np.uint8), cmap= 'gray')
            ax3.imshow((r>.5).detach().squeeze().cpu().numpy().astype(np.uint8), cmap='gray')
    else:
        # only one image has been passed
        f, cols = plt.subplots(number_of_images, 3, figsize=(20,10))
        cols[0].set_title('GT masks')
        cols[1].set_title("Predicted masks")
        cols[2].set_title("Refined masks")
        cols[0].imshow(gt_masks.detach().squeeze().cpu().numpy().astype(np.uint8)*255, cmap='gray') 
        cols[1].imshow((segmented_masks>.5).detach().squeeze().cpu().numpy().astype(np.uint8), cmap= 'gray')
        cols[2].imshow((refined_masks>.5).detach().squeeze().cpu().numpy().astype(np.uint8), cmap='gray')

    f.savefig(path / f'{image_name}.png')
    plt.close()

def unfold_patches(gt, images, predictions, size=128, stride = 128, test = False):
    """Compute patches of current batch RGB images using the unfold method. Patches have dimension "size", default is 128 because images in default settings have 768x768 dimension.
    
    Parameters
    ----------
    - gt: ground truth masks in batch format, having shape (B,1,MASK_SIZE, MASK_SIZE)
    - images: batch of RGB images, shape is (B,3,IMG_SIZE,IMG_SIZE) 
    - predictions: batch of predicted segmentation masks from segmentation model. Nota bene: they are logits so need to threshold them to obtain segm. masks properly
    - size: size of patches, default is 128
    - stride: stride should be equal to size in order to avoid overlapping
    - test: default False, if true means that this function has been called from test_step where labels are not required.
    
    Returns
    -------
    - images_patches: patches in unfolded view, shape is (N*Batch_size, 3, 128, 128) where N is the number of extracted patches from each image, in default settings is 36
    - labels: tensor of shape (N*Batch_size), contains labels associated with each patch. Indexes are coherent, meaning that labels[X] is label of patch images_patches[X].
    """
    channels = images.shape[1] # RGB => 3
    #unfold RGB images in patches
    images_patches = images.unfold(2,size,stride).unfold(3,size,stride).unfold(4,size,stride) 
    images_patches = images_patches.contiguous().view(-1,channels,size,size) # reshape to (36*Batch_size, 3, 128, 128)

    #unfold predicted masks in the same way as RGB images in order to concat them in a 4 channels image
    pred = (predictions > .5).float()
    pred_u = pred.unfold(2,size,stride).unfold(3,size,stride).unfold(4,size,stride)
    pred_u = pred_u.contiguous().view(-1,1,size,size) # channels here is 1, shape will be (36*Batch_size, 1, 128, 128) 
    
    #concat to het 4 channels patches
    images_patches = torch.cat((images_patches, pred_u), dim = 1)
    
    if test is False: 
        #unfold gt masks in the same way as RGB images in order to produce coherent classification labels
        gt_patches = gt.unfold(2,size,stride).unfold(3,size,stride).unfold(4,size,stride)
        gt_patches = gt_patches.contiguous().view(-1,1,size,size) # channels here is 1, shape will be (36*Batch_size, 1, 128, 128) 
        r = torch.sum(torch.sum(gt_patches, dim = 2), dim = 2) # count the number of pixels set to 1 in each patch
        labels = torch.where(r>200,1,0) #NOTE if at least 200 pixels are set to 1 consider patch as positive
        labels = labels.view(labels.shape[0]) # reshape tensor to shape [N,] as expected from CrossEntropyLoss
        return images_patches, labels
    else:
        # call from test_step, avoid label computation because not required
        return images_patches
    
def refine_predictions_unfolding(predictions, labels, size = 128, stride = 128, channels = 1):
    """Refine predicted segmentation masks with labels computed from classifier over RGB images.
    
    Parameters
    ----------
    predictions: tensor with shape [B,1,768,768] where B batch size. Contains segmentation model predicted masks over batch images.
    labels: tensor of shape [N,] contains labels (0/1) predicted from classifier over each patch extracted from RGB images in current batch.
    
    Return
    ------
    refined predictions: Tensor of shape [B,1,768,768]. According to classifier predictions, segmentation masks are refined setting to 0 areas which corresponds to patches classified as 0 (not containing cysts) from classifier.
    """
    batch_size = predictions.shape[0]
    u_pred = predictions.unfold(2,size,stride).unfold(3,size,stride).unfold(4,size,stride) # unfold predicted masks
    unfold_shape = u_pred.size() # save unfolded shape for later (reconstruction after refinement)
    
    u_pred = u_pred.contiguous().view(-1,channels,size,size) # reshaping in order to muliply it with labels for refinement
    labels = labels.reshape(labels.shape[0], 1 , 1, 1) # reshape labels in order to exploit broadcasting

    # refine prediction: labels 1 means "should contain cysts" so predicted area is inaltered, 
    # 0 means "here shouldn't be cysts" so predicted are is refined as black
    refined_mask = u_pred * labels

    # reconstruct batch shape after refinement
    rec = refined_mask.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    rec = rec.permute(0, 1, 4, 2, 6, 3, 5).contiguous() # permute to keep original orientation of mask
    rec = rec.view(batch_size, output_c, output_h, output_w) # reshape as originally in batch format

    return rec #shape is [B,1,768,768] where B: batch size

   


