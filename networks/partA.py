import torch
import numpy as np
from networks.dinknet import DinkNet34
from torchvision import transforms
from PIL import Image
import cv2 as cv
import os
import torchvision.transforms.functional as TF
from utils.loader.dataset import get_folder_img_ids
from torch.autograd import Variable as V

BATCHSIZE_PER_CARD = 4

class TTAFrame:
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        return

    def test_one_img(self, img, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_1(img)
        elif batchsize >= 4:
            return self.test_one_img_2(img)
        elif batchsize >= 2:
            return self.test_one_img_4(img)

    def test_one_img_8(self, img):
        # img = img.transpose(1, 2, 0)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img2 = torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img3 = torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img4 = torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda()

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_4(self, img):
        # img = img.transpose(1, 2, 0)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img2 = torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img3 = torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img4 = torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda()

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_2(self, img):
        # img = img.transpose(1, 2, 0)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = torch.Tensor(img5).cuda()
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = torch.Tensor(img6).cuda()

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_1(self, img):
        img90 = np.array(np.rot90(img))
        # (256, 256, 3)
        # (256, 256, 3)
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = torch.Tensor(img5).cuda()

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        return


def crop_dataset(source_folder, save_folder, image_, mask_):
    """
            Process a dataset by cropping images and labels into 256x256 patches,
            And compute the count of the patches pairs.

            Args:
            image_ (str): Suffix of the input img.
            mask_ (str): Suffix of the corresponding label mask of the img.
            source_folder(str): Dataset's root.
            save_folder(str): Cropped dataset's root.

            Returns:
            list: A list of cropped img patches (tensor : 256 * 256 * 3).
            list: A list of corresponding cropped label patches (tensor : 256 * 256 * 3).
            int: The count of the patches pairs.
    """
    ids = get_folder_img_ids(source_folder, image_, mask_)

    count = 0
    cropped_imgs = []
    cropped_labels = []

    for m in range(len(ids)):
        img = Image.open(f"{source_folder}/{ids[m]}{image_}").convert('RGB')
        label = Image.open(f"{source_folder}/{ids[m]}{mask_}").convert('RGB')

        transform = transforms.Compose(
            [transforms.Resize((1024, 1024)),
             transforms.ToTensor()])

        img = transform(img)
        label = transform(label)
        # torch.Size([3, 1024, 1024])
        # torch.Size([3, 1024, 1024])

        n = 1

        # Crop images and labels into 256 x 256 * 3 patches
        for i in range(0, 1024, 256):
            for j in range(0, 1024, 256):
                cropped_img = img[:, i:i + 256, j:j + 256]
                cropped_label = label[:, i:i + 256, j:j + 256]

                img1 = transforms.functional.to_pil_image(cropped_img)
                lab1 = transforms.functional.to_pil_image(cropped_label)
                img1.save(os.path.join(save_folder, f"{ids[m]}_{n}"+"sat"+image_[-4:]))
                lab1.save(os.path.join(save_folder, f"{ids[m]}_{n}"+"mask"+mask_[-4:]))

                cropped_img = (cropped_img * 255).permute(1, 2, 0)
                cropped_label = (cropped_label * 255).permute(1, 2, 0)
                # torch.Size([256,256,3])
                # torch.Size([256,256,3])
                cropped_imgs.append(cropped_img)
                cropped_labels.append(cropped_label)

                count += 1
                n += 1

    return cropped_imgs, cropped_labels, count


def process_dataset(image_, mask_, source_folder, save_folder, unusual_percent, is_clean):
    """
        Process a dataset by using a pre-trained D-LinkNet34 model for label prediction.

        Args:
        image_ (str): Suffix of the input img.
        mask_ (str): Suffix of the corresponding label mask of the img.
        source_folder(str): Cropped Dataset's root.
        save_folder(str): Processed dataset's root.
        unusual_percent(float): A parameter set for the imprecise model.
        is_clean (bool): A flag indicating whether the dataset is clean or not.

        Returns:
        list: A list of cropped img patches (tensor : 256 * 256 * 3).
        list: A list of corresponding pre-trained label patches (tensor : 256 * 256 * 3).
        float: The labeling rate of the processed dataset.
    """

    ids = get_folder_img_ids(source_folder, image_, mask_)

    cropped_imgs = []
    cropped_labels = []
    unusual_predict = []

    labeled_patches_count = 0
    total_patches_count = 0

    for m in range(len(ids)):
        img = cv.imread(f"{source_folder}/{ids[m]}{image_}")
        label = cv.imread(f"{source_folder}/{ids[m]}{mask_}")

        cropped_img = transforms.ToTensor()(img)
        cropped_label = transforms.ToTensor()(label)
        # torch.Size([3, 256, 256])
        # torch.Size([3, 256, 256])

        cropped_img = cropped_img.permute(1, 2, 0)
        cropped_label = cropped_label.permute(1, 2, 0)
        # torch.Size([256, 256, 3])
        # torch.Size([256, 256, 3])

        total_patches_count += 1

        # using a pre-trained D-LinkNet34 model for label prediction
        if torch.sum(cropped_label) > 0:
            labeled_patches_count += 1
            cv.imwrite(save_folder + f'{ids[m]}' + mask_, label.astype(np.uint8))

        elif not is_clean:
            solver = TTAFrame(DinkNet34)
            solver.load('/public/zjj/public/zjj/jy/work1/weights/small_log01_dinknet34_deepglo.th')
            mask = solver.test_one_img(img)
            mask[mask > 4.0] = 255
            mask[mask <= 4.0] = 0
            mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
            pred_label = torch.tensor(mask)

            if (torch.sum(pred_label) > 0) and (torch.sum(pred_label) < 255 * 255 * 255 * 3 * unusual_percent):
                cv.imwrite(save_folder + f'{ids[m]}' + mask_, mask.astype(np.uint8))
                labeled_patches_count += 1
            else:
                if (torch.sum(pred_label) > 255 * 255 * 255 * 3 * unusual_percent):
                    print("Unusual: " + f'{ids[m]}' + image_)
                    unusual_predict.append(cropped_img)
                mask = np.zeros((256, 256, 3))
                cv.imwrite(save_folder + f'{ids[m]}' + mask_, mask.astype(np.uint8))

            pred_label = torch.tensor(mask)
            cropped_label = torch.round(pred_label).squeeze(0)

        cropped_imgs.append(cropped_img)
        cropped_labels.append(cropped_label)

    # Calculate the labeling rate
    labeling_rate = labeled_patches_count / total_patches_count if total_patches_count > 0 else 0

    return cropped_imgs, cropped_labels, unusual_predict, labeling_rate