import torch
from time import perf_counter

from networks.partC import get_PartC
from networks.partD import get_PartD
from networks.partE import get_PartE
from networks.partF import get_PartF
from networks.vit import ViTEncoder

TEST_BATCH_SIZE = 2

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.part_c = get_PartC()
        self.part_d = get_PartD()
        self.part_e = get_PartE()
        self.part_f = get_PartF()
        return
    
    def train(self, imgs, masks, debug=False):
        a_imgs        = torch.rand(TEST_BATCH_SIZE * 16, 3, 256, 256)
        a_labels      = torch.rand(TEST_BATCH_SIZE * 16, 256, 256)
        a_is_positive = torch.ones(TEST_BATCH_SIZE * 16).long()

        if debug:
            print(f"[MODEL] Part C imgs's shape {a_imgs.shape}")
            print(f"[MODEL] Part C labels's shape {a_labels.shape}")
            print(f"[MODEL] Part C is_positive's shape {a_is_positive.shape}")
        c_loss, c_output = self.part_c.train(a_imgs, a_labels, a_is_positive)
        c_labels = a_labels
        if debug:
            print(f"[MODEL] Part C outputs's shape {c_output.shape}")

        if debug:
            print(f"[MODEL] Part D inputs's shape {c_output.shape}")
        d_loss, d_output = self.part_d.train(c_output.detach(), c_labels)
        if debug:
            print(f"[MODEL] Part D outputs's shape {d_output.shape}")
            
        if debug:
            print(f"[MODEL] Part E inputs's shape {d_output.shape}")
        e_output = self.part_e(d_output)
        if debug:
            print(f"[MODEL] Part E outputs's shape {e_output.shape}")

        LOSS = c_loss + d_loss

        return LOSS
    
    def forward(self):
        return
    
class ViTModel(torch.nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        self.model = ViTEncoder(
            img_size = 1024,
            patch_size = 32,
            in_channels = 3,
            embed_dim = 1024,
            num_heads = 4,
            num_layers = 6,
            mlp_hidden_dim = 1024,
            use_cls = False,
            dropout = 0.5,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()
        return
    
    def train(self, imgs, masks):
        outputs = self.model(imgs)
        self.optimizer.zero_grad()
        loss = self.criterion(outputs, masks)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def forward(self, imgs):
        outputs = self.model(imgs)
        return outputs

class FrameWork:
    def __init__(self, logger, train_loader, test_loader):
        self.model = Model()
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        return

    def _optim_(self, image, pos):
        LOSS = self.model.train(image, pos, debug=True)
        return LOSS

    def _test_(self, CUDA):
        return 1.8

    def train(self, CUDA, EPOCHS, SHOW_STEP, TEST_STEP, SAVE_STEP):
        START = perf_counter()
        STEP = 0

        for EPOCH in range(EPOCHS):
            for image, pos in self.train_loader:
                image = image.cuda() if CUDA else image
                pos = pos.cuda() if CUDA else pos

                LOSS = self._optim_(image, pos)

                if STEP == 0:
                    self.logger.log(f"[INFO] Start training.")

                if STEP % SHOW_STEP == 0:
                    TIME = perf_counter() - START
                    START = perf_counter()
                    self.logger.log(f"[INFO] Epoch:{EPOCH} Step:{STEP} Loss:{LOSS:.6f} Time:{TIME:.2f}s.")

                # if STEP % TEST_STEP == 0:
                #     LOSS = self._test_(CUDA)
                #     self.logger.log(f"[INFO] Epoch:{EPOCH} Step:{STEP} Test-loss:{LOSS:.6f}.")

                if STEP % SAVE_STEP == 0:
                    self.logger.save(self.model, STEP)
                    self.logger.log(f"[INFO] Epoch:{EPOCH} Step:{STEP} Model save as _{STEP}_.pth.")

                STEP = STEP + 1

        self.logger.log(f"[INFO] Finished training.")
        return
