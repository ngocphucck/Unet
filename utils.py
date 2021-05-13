import os
import json
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm


os.chdir('/home/doanphu/Documents/Code/Practice/Unet')


def create_mask_image(annotation_path='data/annotations/instances_train.json',
                      save_path='data/mask_train/'):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        print(f'Create {save_path}')

    coco = COCO(annotation_path)
    imgIds = coco.getImgIds(catIds=[1])

    for i in tqdm(range(len(imgIds))):
        img = coco.loadImgs(imgIds[i])[0]
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=[1], iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        mask = coco.annToMask(anns[0])
        for j in range(1, len(anns)):
            mask += coco.annToMask(anns[j])

        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

        save_mask_path = os.path.join(save_path, img['file_name'].split('/')[-1])
        mask.save(save_mask_path)


if __name__ == '__main__':
    create_mask_image(annotation_path='data/annotations/instances_test.json',
                      save_path='data/mask_test')
    pass
