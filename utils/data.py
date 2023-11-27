import torch
import numpy as np
from networks.dinknet import DinkNet34
from torchvision import transforms
from PIL import Image

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
        img = img.transpose(1, 2, 0)
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
        img = img.transpose(1, 2, 0)
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
        img = img.transpose(1, 2, 0)
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


def process_dataset(img_img, label_img, is_clean):
    """
    Process a dataset by cropping images and labels into 256x256 patches, and compute the labeling rate.
    Using a pre-trained D-LinkNet34 model for label prediction.

    Args:
    img_img (str): img to the input img.
    label_img (str): img to the corresponding label mask of the img.
    is_clean (bool): A flag indicating whether the dataset is clean or not.

    Returns:
    list: A list of cropped img patches (tensor : 256 * 256 * 3).
    list: A list of corresponding cropped label patches (tensor : 256 * 256 * 3).
    float: The labeling rate of the processed dataset.
    """

    img = Image.open(img_img).convert('RGB')
    label = Image.open(label_img).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    img = transform(img)
    label = transform(label)
    # shape:
    # torch.Size([3, 1024, 1024])
    # torch.Size([1, 1024, 1024])

    cropped_imgs = []
    cropped_labels = []

    labeled_patches_count = 0
    total_patches_count = 0

    # Crop images and labels into 256 x 256 * 3 patches
    for i in range(0, 1024, 256):
        for j in range(0, 1024, 256):
            cropped_img = img[:, i:i + 256, j:j + 256]
            cropped_label = label[:, i:i + 256, j:j + 256]

            cropped_img = cropped_img.permute(1, 2, 0)
            cropped_label = cropped_label.permute(1, 2, 0)
            # shape:
            # torch.Size([256, 256, 3])
            # torch.Size([256, 256, 3])

            total_patches_count += 1

            if torch.sum(cropped_label) > 0:
                labeled_patches_count += 1
            elif not is_clean:
                solver = TTAFrame(DinkNet34)
                solver.load('./../pretrained/DLinkNet34.th')

                mask = solver.test_one_img(cropped_img.numpy())
                mask[mask > 1.0] = 255
                mask[mask <= 1.0] = 0
                mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]],
                                      axis=2)

                pred_label = torch.tensor(mask)
                cropped_label = torch.round(pred_label).squeeze(0)
                if torch.sum(cropped_label) > 0:
                    labeled_patches_count += 1

            # print('---')
            # print(cropped_img.shape)
            # print(cropped_label.shape)
            # print('---')

            cropped_imgs.append(cropped_img)
            cropped_labels.append(cropped_label)

    # Calculate the labeling rate
    labeling_rate = labeled_patches_count / total_patches_count if total_patches_count > 0 else 0

    return cropped_imgs, cropped_labels, labeling_rate
