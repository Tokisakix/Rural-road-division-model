import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

class NewLabelGenerator:
    def __init__(self):
        pass

    def process_cam(self, cam_images):
        '''
        Process CAM to generate binary labels.

        Args:
        - cam_images (list): List of CAM images, each of size (256, 256, 3).

        Returns:
        - processed_labels (list): List of binary labels, each of size (256, 256, 3).
        '''
        processed_labels = []

        for cam_image in cam_images:
            resized_cam = cv2.resize(cam_image, (256, 256))  # CAM: (256, 256, 3)

            # summing the three channels before binarizing
            summed_channels = np.sum(resized_cam, axis=2)

            # binarization
            # Set to white (255) if greater than threshold, else black (0)
            threshold_value = 255 / 2
            binary_cam = np.zeros_like(summed_channels, dtype=np.uint8)
            binary_cam[summed_channels >= threshold_value * 3] = 255

            # stack to form a three-channel binary image
            binary_cam = np.stack([binary_cam] * 3, axis=-1)

            print(binary_cam)

            processed_labels.append(binary_cam)

        return processed_labels

    def visualize_label(self, label):
        '''
        Help visualize the labels.
        '''
        plt.imshow(cv2.cvtColor(label, cv2.COLOR_BGR2GRAY), cmap='gray')
        plt.axis('off')
        plt.title('Generated Label')
        plt.show()

    def assemble(self, processed_labels):
        '''
        Assemble the 16 small (256, 256, 3) labels to a greater one (1024, 1024, 3)
        '''
        assembled_label = np.zeros((1024, 1024, 3), dtype=np.uint8)

        row = col = 0
        for idx, label in enumerate(processed_labels):
            row = (idx // 4) * 256
            col = (idx % 4) * 256
            assembled_label[row:row+256, col:col+256, :] = label

        return assembled_label

# # Example usage
# new_label_generator = NewLabelGenerator()
# cam_images = [np.random.rand(256, 256, 3) * 255 for _ in range(16)]  # Using 16 randomly generated CAM images as an example
#
# processed_labels = new_label_generator.process_cam(cam_images)
# for label in processed_labels:
#     new_label_generator.visualize_label(label)
#
# new_label = new_label_generator.assemble(processed_labels)
# new_label_generator.visualize_label(new_label)

class PartE(torch.nn.Module):
    def __init__(self, patch_num) -> None:
        super(PartE, self).__init__()
        self.patch_num = patch_num
        return
    
    def forward(self, imgs):
        imgs = imgs.permute(0, 2, 3, 1)
        (b, w, h, c) = imgs.shape
        imgs = imgs.reshape(b // self.patch_num, w * int(self.patch_num ** 0.5), h * int(self.patch_num ** 0.5), c)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs
    
def get_PartE():
    parte = PartE(
        patch_num = 16,
    )
    return parte

if __name__ == "__main__":
    inputs = torch.rand(32, 3, 256, 256)

    print("inputs.shape", inputs.shape)
    parte = get_PartE()
    outputs = parte(inputs)
    print("outputs.shape", outputs.shape)