import argparse
from torchvision.transforms import Compose, Resize, ToTensor
import cv2
import torch


from model import Unet


def predict(image_path,
            checkpoint_path,
            save_path):
    model = Unet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    size = image.shape
    image = Compose([
        ToTensor(),
        Resize((512, 512)),
    ])(image)
    image = image.unsqueeze(0)

    mask = model(image)[0]
    mask[mask < 0.5] = 0
    mask[mask > 0.5] = 255
    mask = Resize(size)(mask)
    mask = mask.detach().numpy()

    cv2.imwrite('result.png', mask[0])
    pass


def get_args():
    parser = argparse.ArgumentParser(
        description='Predict x-ray image'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        default='',
        help='path to image (default: None)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='',
        help='path to the checkpoint (default: None)'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='',
        help='path to save image (default: None)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    predict(args.image_path, args.weights, args.save_path)
    pass
