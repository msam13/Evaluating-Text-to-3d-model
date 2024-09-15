import os
import sys
import glob
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path

import sys
sys.path.append('../../')
import utils.utils as utils
import projects.dsine.config as config
from utils.projection import intrins_from_fov, intrins_from_txt
fx = 400.0
fy = 400.0
cx = 400.0
cy = 400.0
intrincis = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


# Path to the folder containing subdirectories
results_folder = pathlib.Path("/home/screwdriver/Experiments/DSINE/data/results")

# Loop through all directories in the folder
for subdir in results_folder.iterdir():
    if subdir.is_dir():  # Check if it's a directory
        print(f"Processing folder: {subdir}")
        # Your processing code here
        parent_dir=subdir
        device = torch.device('cuda')
        args = config.get_args(test=True)
        assert os.path.exists(args.ckpt_path)

        if args.NNET_architecture == 'v00':
            from models.dsine.v00 import DSINE_v00 as DSINE
        elif args.NNET_architecture == 'v01':
            from models.dsine.v01 import DSINE_v01 as DSINE
        elif args.NNET_architecture == 'v02':
            from models.dsine.v02 import DSINE_v02 as DSINE
        elif args.NNET_architecture == 'v02_kappa':
            from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
        else:
            raise Exception('invalid arch')

        model = DSINE(args).to(device)
        model = utils.load_checkpoint(args.ckpt_path, model)
        model.eval()

        img_paths = glob.glob(f'{parent_dir}/rgb/*.png') 
        img_paths.sort()   

        os.makedirs(f'{parent_dir}/output/', exist_ok=True)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with torch.no_grad():
                for img_path in img_paths:
                    print(img_path)
                    ext = os.path.splitext(img_path)[1]
                    img = Image.open(img_path).convert('RGB')
                    img = np.array(img).astype(np.float32) / 255.0
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

                    # pad input
                    _, _, orig_H, orig_W = img.shape
                    lrtb = utils.get_padding(orig_H, orig_W)
                    img = F.pad(img, lrtb, mode="constant", value=0.0)
                    img = normalize(img)

                    # get intrinsics
                    intrins = torch.from_numpy(np.array([intrincis]) ).to(device)
                    print(intrins)
                    intrins[:, 0, 2] += lrtb[0]
                    intrins[:, 1, 2] += lrtb[2]

                    pred_norm = model(img, intrins=intrins)[-1]
                    pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

                    # save to output folder
                    # NOTE: by saving the prediction as uint8 png format, you lose a lot of precision
                    # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files
                    imagename = Path(img_path).stem
                    mask = np.load(f'{parent_dir}/mask/{imagename}.npy')

                    DSINE_out_folder = f"{parent_dir}/DINE_output/"
                    os.makedirs(DSINE_out_folder, exist_ok=True)
                    target_path = f"{DSINE_out_folder}/{imagename}.png"

                    pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
                    pred_norm[0, ...] = pred_norm[0, ...] * ~mask

                    DSINE_npy_folder = f"{parent_dir}/DINE_npy_OUTPUT/"
                    os.makedirs(DSINE_npy_folder, exist_ok=True)
                    npyname = f"{DSINE_npy_folder}/{imagename}.npy"
                    np.save(npyname, pred_norm[0, ...])

                    
                    pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)
                    img = pred_norm[0,...]
                    # img = img * ~mask
                    im = Image.fromarray(img)
                    im.save(target_path)