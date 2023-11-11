from time import perf_counter

class FrameWork:
    def __init__(self, model, logger, train_loader, test_loader, optimizer, criterion):
        self.model = model
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        return

    def _optim_(self, image, pos, neg):
        return 1.6

    def _test_(self, CUDA):
        return 1.8

    def train(self, CUDA, EPOCHS, SHOW_STEP, TEST_STEP, SAVE_STEP):
        START = perf_counter()
        STEP = 0

        for EPOCH in range(EPOCHS):
            for image, pos, neg in self.train_loader:
                image = image.cuda() if CUDA else image
                pos = pos.cuda() if CUDA else pos
                neg = neg.cuda() if CUDA else neg

                LOSS = self._optim_(image, pos, neg)

                if STEP == 0:
                    self.logger.log(f"[INFO] Start training.")

                if STEP % SHOW_STEP == 0:
                    TIME = perf_counter() - START
                    START = perf_counter()
                    self.logger.log(f"[INFO] Epoch:{EPOCH} Step:{STEP} Loss:{LOSS:.6f} Time:{TIME:.2f}s.")

                if STEP % TEST_STEP == 0:
                    LOSS = self._test_(CUDA)
                    self.logger.log(f"[INFO] Epoch:{EPOCH} Step:{STEP} Test-loss:{LOSS:.6f}.")

                if STEP % SAVE_STEP == 0:
                    self.logger.save(self.model, STEP)
                    self.logger.log(f"[INFO] Epoch:{EPOCH} Step:{STEP} Model save as _{STEP}_.pth.")

                STEP = STEP + 1
        
        self.logger.log(f"[INFO] Finished training.")
        return