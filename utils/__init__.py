import json
import os

CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config

class LogPrinter:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(log_path, "w", encoding="utf-8") as log:
            pass
        return
    
    def log(self, log_content):
        with open(self.log_path, "a+", encoding="utf-8") as log:
            log.write(log_content + "\n")
        return