import faulthandler
faulthandler.enable()

import os
import time

import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    """
    :return:
    input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode,
    strength, scale, seed, eta
    """
    desc = "Transform reality video to anime video by using CPU torch <control NET>"

    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--img1', type=str, required=True, help='img1 file')
    parser.add_argument('--img2', type=str, required=True, help='img2 file')
    parser.add_argument('--exp', default=4, type=int)
    parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
    parser.add_argument('--rthreshold', default=0.02, type=float,
                        help='returns image when actual ratio falls in given range threshold')
    parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
    parser.add_argument('--model', dest='modelDir', type=str, default='train_log',
                        help='directory with trained model files')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    kwargs = {
        "imgs": [args.img1, args.img2],
        "exp": args.exp,
        "ratio": args.ratio,
        "modelDir": args.modelDir
    }
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model

                model = Model()
                model.load_model(kwargs["modelDir"], -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model

                model = Model()
                model.load_model(kwargs["modelDir"], -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model

            model = Model()
            model.load_model(kwargs["modelDir"], -1)
            print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model

        model = Model()
        model.load_model(kwargs["modelDir"], -1)
        print("Loaded ArXiv-RIFE model")
    model.eval()
    model.device()

    if kwargs["imgs"][0].endswith('.exr') and kwargs["imgs"][1].endswith('.exr'):
        img0 = cv2.imread(kwargs["imgs"][0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img1 = cv2.imread(kwargs["imgs"][1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)

    else:
        img0 = cv2.imread(kwargs["imgs"][0], cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(kwargs["imgs"][1], cv2.IMREAD_UNCHANGED)
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    if kwargs["ratio"]:
        img_list = [img0]
        img0_ratio = 0.0
        img1_ratio = 1.0
        if kwargs["ratio"] <= img0_ratio + args.rthreshold / 2:
            middle = img0
        elif kwargs["ratio"] >= img1_ratio - args.rthreshold / 2:
            middle = img1
        else:
            tmp_img0 = img0
            tmp_img1 = img1
            for inference_cycle in range(args.rmaxcycles):
                middle = model.inference(tmp_img0, tmp_img1)
                middle_ratio = (img0_ratio + img1_ratio) / 2
                if kwargs["ratio"] - (args.rthreshold / 2) <= middle_ratio <= kwargs["ratio"] + (args.rthreshold / 2):
                    break
                if kwargs["ratio"] > middle_ratio:
                    tmp_img0 = middle
                    img0_ratio = middle_ratio
                else:
                    tmp_img1 = middle
                    img1_ratio = middle_ratio
        img_list.append(middle)
        img_list.append(img1)
    else:
        img_list = [img0, img1]
        for i in range(kwargs['exp']):
            print(f"exp:{i}")
            tmp = []
            for j in range(len(img_list) - 1):
                mid = model.inference(img_list[j], img_list[j + 1])
                tmp.append(img_list[j])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp

    if not os.path.exists('output'):
        os.mkdir('output')
    for i in range(len(img_list)):
        print(f"img_list:{i}")
        if kwargs["imgs"][0].endswith('.exr') and kwargs["imgs"][1].endswith('.exr'):
            cv2.imwrite('output/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w],
                        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        else:
            print('output/img{}.png'.format(i))
            cv2.imwrite('output/img{}.png'.format(i),
                        (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

