import torch
import datetime
import os

class LogPrinter:
    def __init__(self, LOG_ROOT, SAVE_NUM):
        self.log_root = LOG_ROOT
        self.log_path = LOG_ROOT + "/log.txt"
        self.save_num = SAVE_NUM
        self.model_num = 0
        self.model_name = []
        with open(self.log_path, "w", encoding="utf-8") as log:
            pass
        return
    
    def log(self, log_content):
        log_content = f"{datetime.datetime.now()} " + log_content
        with open(self.log_path, "a+", encoding="utf-8") as log:
            log.write(log_content + "\n")
        print(log_content)
        return
    
    def save(self, model, step):
        SAVE_PATH = self.log_root + f"/_{step}_.pth"
        torch.save(model, SAVE_PATH)
        print("sucess",SAVE_PATH)
        self.model_num += 1
        self.model_name.append(SAVE_PATH)
        if self.model_num > self.save_num:
            os.remove(self.model_name[0])
            model_name = self.model_name[0].split("/")[-1]
            del self.model_name[0]
            self.log(f"[INFO] Remove {model_name}.")
        return
