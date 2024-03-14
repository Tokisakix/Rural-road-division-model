import torch
from time import perf_counter, time

from networks.partC import get_PartC
from networks.partD import get_PartD
from networks.partE import get_PartE
from networks.partF import get_PartF
from networks.vit import ViTEncoder
from networks.dinknet import DinkNet34


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.part_c = get_PartC()
        self.part_d = get_PartD()
        self.part_e = get_PartE()
        self.part_f = get_PartF()
        return

    def train(self, imgs, masks):
        # is_positive = torch.ones(imgs.shape[0]).long().cuda()

        c_loss, c_output = self.part_c.train(imgs, masks)

        c_labels = masks

        d_loss, d_output = self.part_d.train(c_output.detach(), c_labels)

        LOSS = c_loss + d_loss

        # e_output = self.part_e(d_output)
        return LOSS

    
    def forward(self, imgs):

        c_output = self.part_c(imgs)

        d_output = self.part_d(c_output)

        # e_output = self.part_e(d_output)
        return d_output
    
class ViTModel(torch.nn.Module):
    def __init__(self, logger, train_loader, test_loader, CUDA, Devices):
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
        self.model = self.model.cuda()
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)
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
    
class OtherPredict(torch.nn.Module):
    def __init__(self, model, criterion, optimizer, logger, train_loader, test_loader, CUDA, Devices):
        super(OtherPredict, self).__init__()
        self.module = model.cuda() if CUDA else model
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)
        self.loss = criterion.cuda() if CUDA else criterion
        return
    
    def train(self, CUDA, EPOCHS, SHOW_STEP, TEST_STEP, SAVE_STEP):
        START = perf_counter()
        STEP = 0

        for epoch in range(EPOCHS):
            # 训练
            # running_loss = 0  # 所有数据误差的总和
            print(f"-------------------------EPOCH: {epoch}-------------------------")

            for data in self.train_loader:
                imgs, masks = data
                if CUDA:
                    print("Using GPU")
                    imgs = imgs.reshape(-1, 3, 256, 256).cuda()
                    masks = masks.reshape(-1, 1, 256, 256).cuda()

                input = imgs
                output = self.module(input)
                result_loss = self.loss(output, masks)

                if STEP == 0:
                    self.logger.log(f"[INFO] Start training. Input.shape:{input.shape}")

                self.optimizer.zero_grad()
                result_loss.backward()
                self.optimizer.step()

                if STEP % SHOW_STEP == 0:
                    TIME = perf_counter() - START
                    START = perf_counter()
                    self.logger.log(f"[INFO] Epoch:{epoch} Step:{STEP} Loss:{result_loss:.6f} Time:{TIME:.2f}s.")

                if STEP % SAVE_STEP == 0:
                    self.logger.save(self.module, STEP)
                    self.logger.log(f"[INFO] Epoch:{epoch} Step:{STEP} Model save as _{STEP}_.pth.")

                STEP += 1
            self.logger.log(f"[INFO] Finished training.")
        return
    def forward(self, imgs):
        outputs = self.module(imgs)
        return outputs

class FrameWork:
    def __init__(self, logger, train_loader, test_loader, CUDA, Devices):
        self.model = Model().cuda() if CUDA else Model()
        # if CUDA:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=Devices, output_device=0)
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        return

    def _optim_(self, image, pos):
        LOSS = self.model._optim_(image, pos)
        return LOSS

    def _test_(self, CUDA):
        pass

    def train(self, CUDA, EPOCHS, SHOW_STEP, TEST_STEP, SAVE_STEP):
        START = perf_counter()
        STEP = 0

        for EPOCH in range(EPOCHS):
            for image, pos in self.train_loader:
                image = image.reshape(-1, 3, 256, 256).cuda() if CUDA else image.reshape(-1, 3, 256, 256)
                pos = pos.reshape(-1, 1, 256, 256).cuda() if CUDA else pos.reshape(-1, 1, 256, 256)

                LOSS = self.model.train(image, pos)

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