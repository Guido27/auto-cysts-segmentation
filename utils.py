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

def extract_patches(gt, pred, image, cutoff=0, p_size = 64, padding_default = 20):
    """
    This functions extract patches from RGB image accoding to predicted segmentation mask. Each extracted patch has an assigned label: 0 if the segmented area does't correspond with a segmentation area in gt mask,
    1 if even a signle pixel in segmented area correspond to a segmented one in gt mask. Takes in input the entire batch segmented from segmentation model having shape (B,1,IMG_SIZE,IMG_SIZE), returns 
    a list of all patches extracted in all images in batch with corresponding labels, coordinates and number of extracted patch for each image.

    Parameters
    ----------
    - gt: ground truth masks in batch format, having shape (B,1,MASK_SIZE, MASK_SIZE)
    - pred: predicted segmentation masks by segmentation model, shape is the same as gt. Pred contains logits, so to obtain the corresponding mask needs to be thresholded correctly.
    - image: batch of images, shape is (B,3,IMG_SIZE,IMG_SIZE) 
    
    Return
    ------
    Tuple of (t,labels,coordinates,patch_each_image), where:
    - t: tensor of shape (N, 3, p_size, p_size) containing all extracted patches according to predictions.
    - labels: tensor of shape N which contains labels for each extracted patch in t. 
    - coordinates: list of tuples (x,y,w,h) containing coordinates of extracted paches (which are the same in both predicted mask and image).
    - patch_each_image: list containing the total number of extracted patches from each image in batch. A number for each image, so lenght will be equal to B 

    Where B is batch size, N is number of extracted patches. Indexes are coherent, meaning that patch in t[X] has label labels[X] and it's coordinates are stored in tuple coordinates[X].
    The total number of extracted patches from image with index 0 in batch will be stored in patch_each_image[0].
    """ 

    t = torch.empty((1, 3, p_size, p_size), dtype=torch.float32) #initialize return tensor of tensors
    coordinates = []
    patch_each_image = [] #contains the total number of extracted patches from each image
    labels = torch.empty((1)).type(torch.LongTensor).cuda() # LongTensor required dtype for CrossEntropyLoss
    labels.requires_grad = False
  
    for m,p,i in zip(gt, pred, image):
      # m GT mask, i image, p segmentation predictions from model  
      m = m.detach().squeeze().cpu().numpy().astype(np.uint8)
      predicted = (p>0.5).detach().squeeze().cpu().numpy().astype(np.uint8) # get segmentation model predicted mask from current predicted logits
      i = i.detach().permute(1, 2, 0).cpu().numpy() # permute image as expected from following code
      
      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
      #### compute NEGATIVE PATCHES for current image
      negative_counter = 0  
      gt_contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      pred_contours, _ = cv2.findContours(predicted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
      gt_contours = tuple([c for c in gt_contours if c.size > 4 and cv2.contourArea(c)>cutoff])
      pred_contours = tuple([c for c in pred_contours if c.size > 4 and cv2.contourArea(c)>cutoff])
      pred_seps = tuple([csr_matrix(cv2.fillPoly(np.zeros_like(m), pts=[c], color=(1))) for c in pred_contours])
      sparse_gt = csr_matrix(m)
      for single_pred, c in zip(pred_seps, pred_contours):
          if not single_pred.multiply(sparse_gt).count_nonzero(): # wrong cyst
            # add coordinates to list of coordinates
            x,y,w,h = cv2.boundingRect(c)
            coordinates.append((x,y,w,h)) 
            #extract patch and save in t
            p = padding_default 
            while True:
              crop = i[(y-p):(y+h+p), (x-p):(x+w+p)]
              if (crop.shape[0] != 0 and crop.shape[1] != 0):
                break
              else:
                 p = p-1    
            resized = cv2.resize(crop, (p_size,p_size), interpolation = cv2.INTER_CUBIC) # resize cropped portion
            t = torch.cat((t,image_to_tensor(resized).unsqueeze(0)), 0)
            negative_counter = negative_counter + 1 
      # compute labels for extracted negative patches
      labels = torch.cat((labels,torch.zeros((negative_counter)).type(torch.LongTensor).cuda()))    
      
      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
      
      #### compute POSITIVE PATCHES for current image
      positive_counter = 0
      #threshold predicted mask in order to extract contours of cysts
      _, thresh = cv2.threshold(predicted*255, 127, 255, 0) #mask * 255 because threshold expects to have integer values between 0 and 255
      # find contours of segmented areas in thresholded image
      contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
      #iterate over segmented areas contours to find ones with cysts in it
      for k in range(len(contours)):
    
        cnt = contours[k]
        x,y,w,h = cv2.boundingRect(cnt)

        # if a segmented area has an intersection with a segmented portion in gt mask: extract it (with padding) as positive patch...
        if m[(y):(y+h), (x):(x+w)].any():  
          # ... and add coordinates to coordinates list
          coordinates.append((x,y,w,h))
          # avoid that cysts with no space for padding cause errors: get cyst with lower padding
          p = padding_default
          while True:
            crop = i[(y-p):(y+h+p), (x-p):(x+w+p)]
            if crop.shape[0] != 0 and crop.shape[1] != 0:
              break
            else:
               p = p-1  
          resized = cv2.resize(crop, (p_size,p_size), interpolation = cv2.INTER_CUBIC) # resize cropped portion
          t = torch.cat((t,image_to_tensor(resized).unsqueeze(0)), 0)
          positive_counter = positive_counter + 1
    
      # compute labels for extracted positive patches
      labels = torch.cat((labels,torch.ones((positive_counter)).type(torch.LongTensor).cuda()))
    
      # compute total number of extracted patches from current image and add it to list
      patch_each_image.append(positive_counter + negative_counter)

    #exclude always the first empty tensor declared with torch.empty
    return t[1:, :, :, :].cuda(), labels[1:], coordinates, patch_each_image

# this function has been incorporated in extract_patches 
def extract_wrong_cysts(gt, pred, image, cutoff=0, p_size = 64, padding_default = 20):
  """Extract wrong segmented areas in segmentation model predicted mask. 
  A segmented area in a prediction is considered wrong when the corresponding area in the ground truth segmentation mask is totally black.
  If a segmented area is wrong, its coordinates are exploited to extract from RGB image the corresponding patch with, if possible, a passing of "padding_default".
  Each patch is resized to a (p_size * p_size) dimension, default p_size is 64.
  
  Parameters
  ----------
  gt: ground truth mask
  pred: prediction from segmentation model after thresholding, basically (logits>0.5)
  image: RGB image corresponding to segmentation prediction and mask

  Return
  ------
  t: Tensor oh shape (N*C*p_size*p_size) where N is the number of wrong/negative extracted patches 
  w_cysts: A list of wrong patches coordinates in the following format: (x,y,w,h). 
  
  w_cysts indexes are compatible with first dimension of t, example:
  t[X,:,:,:] is patch X and its cordinates are saved in w_cysts[X]
  """
  t = torch.empty((1, 3, p_size, p_size), dtype=torch.float32) #initialize return tensor of tensors
  w_cysts = []
  gt_contours, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  pred_contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  gt_contours = tuple([c for c in gt_contours if c.size > 4 and cv2.contourArea(c)>cutoff])
  pred_contours = tuple([c for c in pred_contours if c.size > 4 and cv2.contourArea(c)>cutoff])
  pred_seps = tuple([csr_matrix(cv2.fillPoly(np.zeros_like(gt), pts=[c], color=(1))) for c in pred_contours])
  sparse_gt = csr_matrix(gt)
  for single_pred, c in zip(pred_seps, pred_contours):
      if not single_pred.multiply(sparse_gt).count_nonzero(): # wrong cyst
       
        # add coordinates to list of coordinates
        x,y,w,h = cv2.boundingRect(c)
        w_tuple = (x,y,w,h)
        w_cysts.append(w_tuple)

        #extract patch and save in t
        p = padding_default
        while True:
          crop = image[(y-p):(y+h+p), (x-p):(x+w+p)]
          if (crop.shape[0] != 0 and crop.shape[1] != 0):
            break
          else:
             p = p-1
        resized = cv2.resize(crop, (p_size,p_size), interpolation = cv2.INTER_CUBIC) # resize cropped portion
        t = torch.cat((t,image_to_tensor(resized).unsqueeze(0)), 0)

  return t[1:, :, :, :] ,w_cysts #exclude the first empty tensor declared with torch.empty

# this function has been incorporated in extract_patches
def extract_real_cysts(gt_mask, pred_mask, image, p_size=64, padding_default=20):
    """Extract from RGB image detected cysts. 
    In order to define if a segmented element is a true cyst the ground truth mask is used: if a segmented object in the prediction mask has even just a single
    pixel in common with a cyst segmented in the groud truth mask is extracted as positive patch for the classifier.
    Each patch will be resized to 64x64 dimension.
    
    Parameters
    ----------
    gt_mask: groud truth segmentation mask
    pred_mask: segmentation mask predicted from segmentation model, passed one is a float tensor converted to numpy array with np.uint8 dtype.
    image: Original RGB image from which positive patches have to be extracted, expected to have shape (H*W*C), C should be 3 because of RGB images
    
    Return
    -------
    t: Tensor oh shape (N*C*p_size*p_size) where N is the number of positive patches extracted
    l: list of coordinates of extracted positive patches"""
    
    t = torch.empty((1, 3, p_size, p_size), dtype=torch.float32) #initialize return tensor of tensors
    l = []
    #threshold predicted mask in order to extract contours of cysts
    ret, thresh = cv2.threshold(pred_mask*255, 127, 255, 0) #mask * 255 because threshold expects to have integer values between 0 and 255
    # find contours of segmented areas in thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #iterate over segmented areas contours to find ones with cysts in it
    for k in range(len(contours)):
    
      cnt = contours[k]
      x,y,w,h = cv2.boundingRect(cnt)
      
      # if a segmented area has an intersection with a segmented cyst in gt mask extract it (with padding) as positive patch
      if gt_mask[(y):(y+h), (x):(x+w)].any():  
        # add coordinates to positive coordinates list
        l.append((x,y,w,h))
        # avoid that cysts with no space for padding cause errors: get cyst with lower padding
        p = padding_default
        while True:
          crop = image[(y-p):(y+h+p), (x-p):(x+w+p)]
          if crop.shape[0] != 0 and crop.shape[1] != 0:
            break
          else:
             p = p-1

        resized = cv2.resize(crop, (p_size,p_size), interpolation = cv2.INTER_CUBIC) # resize cropped portion
        t = torch.cat((t,image_to_tensor(resized).unsqueeze(0)), 0)
    
    return t[1:, :, :, :],l #exclude the first empty tensor declared with torch.empty in t


def extract_segmented_cysts_test_time(prediction, image, p_size = 64, padding_default = 20):
    """Extract patches from image based on segmented areas in prediction using contours over predicted segmented mask (segmented mask = (logits > 0.5)).
       Returns a tensor of patches that the classifier have to classify at test time.
       
       Parameters
       ----------
       prediction: logits from segmentation model after tresholding (prediction > 0.5)
       image: RGB image associated with prediction
       
       Return
       ------
       t: tensor of patches (tenso of tensors) of shape (N*C*p_size*p_size) where N is the number of patches extracted
       l: list of coordinates of extracted patches
       """
    t = torch.empty((1, 3, p_size, p_size), dtype=torch.float32) #initialize return tensor of tensors
    l = []
    #threshold predicted mask in order to extract contours of cysts
    ret, thresh = cv2.threshold(prediction*255, 127, 255, 0) #mask * 255 because threshold expects to have integer values between 0 and 255
    # find contours of segmented areas in thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #iterate over segmented areas contours 
    for k in range(len(contours)):
    
        cnt = contours[k]
        x,y,w,h = cv2.boundingRect(cnt)

        # add coordinates to coordinates list
        l.append((x,y,w,h))
        # avoid that cysts with no space for padding cause errors: get cyst with lower padding
        p = padding_default
        while True:
          crop = image[(y-p):(y+h+p), (x-p):(x+w+p)]
          if crop.shape[0] != 0 and crop.shape[1] != 0:
            break
          else:
             p = p-1
        resized = cv2.resize(crop, (p_size,p_size), interpolation = cv2.INTER_CUBIC) # resize cropped portion
        t = torch.cat((t,image_to_tensor(resized).unsqueeze(0)), 0)

    return t[1:, :, :, :],l #exclude the first empty tensor declared with torch.empty in t

def refine_predicted_masks(logits,coordinates,patch_each_image,predicted_labels):
    """
    Info
    ----
    Function that refine predictions in current batch performed by segmentation model.

    Parameters
    ----------
    logits: logits predicted from segmentation model which receive the entire batch in input and output predictions of same shape. Shape is (B,1,1024,1024) where B is Batch Size.
    coordinates: list of N coordinates, each one is associated with a patch. Represents the coordinates of patches in both image and predicted mask from segmentation model
    patch_each_image: List of lenght equal to batch size, contains in each position the number of patches extracted from each batch image. For example z[3] contains the total number of patches extracted from image/logit of index 3 in current batch.
    predicted_labels: tensor of shape (N) containing predictions over each patch from classifier. 1 if "Cyst" 0 if "Not Cyst"   
    Return
    ------
    T: tensor of shape (logits.shape) which contains refined predictions logits according to patches classification performed by classifier: patches classified as negative are removed from predicted logits  
    """ 
    T = torch.empty((logits.shape)).cuda()
    min_index = 0 #set first min_index to -1, min_index is the index of     
    
    for index in range(logits.shape[0]):
        
        #get current logit
        current_prediction = logits[index].detach().clone()   
        # get number of extracted patches from current prediction
        max_index = patch_each_image[index]   
        # get classifier predictions of patches extracted from current prediction as tensor to exploit indexing
        p = predicted_labels[min_index:min_index+max_index]  
        # get coordinates of patches in p, also here as tensor to exploit indexing
        c = torch.as_tensor(coordinates[min_index:min_index+max_index]).cuda() # cuda because p will be on GPU
        # keep only coordinates of patches classified as not cysts from classifier 
        c = c[p==0]  
        # compute mask of ones with shape equal to current_prediction and set to 0 sections overlapping patches predicted as false
        erasing_mask = torch.ones(current_prediction.shape).cuda()
        erasing_mask.requires_grad = False # debug
        for c_tensor in c:
            # c_tensor has 4 elements: [x,y,w,h]
            x = int(c_tensor[0])
            y = int(c_tensor[1])
            w = int(c_tensor[2])
            h = int(c_tensor[3])
            erasing_mask[0, (y):(y+h), (x):(x+w)] = torch.zeros((1,h,w))    
        # refine mask erasing patches predicted as false with erasing_mask matrix multiplication and save computed refined mask in T (T[index] will contain refined version of logits[index])
        T[index] = current_prediction*erasing_mask    
        #update min_index
        min_index = max_index 

    return T
        
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

def unfold_patches(gt,images, size=128, stride = 128, test = False):
    """Compute patches of current batch RGB images using the unfold method. Patches have dimension "size", default is 128 because images in default settings have 768x768 dimension.
    
    Parameters
    ----------
    - gt: ground truth masks in batch format, having shape (B,1,MASK_SIZE, MASK_SIZE)
    - image: batch of RGB images, shape is (B,3,IMG_SIZE,IMG_SIZE) 
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

    if test is False: 
        #unfold in the same way gt masks in order to produce classification labels
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
    # TODO completare documentazione
    # predictions: tensore con shape [B,1,768,768] dove B batch size
    # labels: tensore di shape [N,] contenente le labels (0 e 1) predette dal classificatore su ciascuna patch dell'immgine RGB
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

   


