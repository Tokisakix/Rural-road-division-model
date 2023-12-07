import json
import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    def __init__(self, input_token_size, output_channel_nums, output_img_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(input_token_size, 128 * 16 * 16)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channel_nums, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.output_img_size = output_img_size

    def forward(self, tokens):
        out = self.fc(tokens)
        out = out.view(-1, 128, 16, 16)
        out = self.conv_layers(out)
        out = nn.functional.interpolate(out, size=(
        self.output_img_size, self.output_img_size))
        return out


class PartD(nn.Module):
    def __init__(self, model, optimizer, criterion, learning_rate):
        super(PartD, self).__init__()
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion
        return

    def train(self, inputs, targets):
        outputs = self.model(inputs)
        self.optimizer.zero_grad()
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item(), outputs

    def forward(self, tokens):
        out = self.model(tokens)
        return out


TEST_BATCH_SIZE = 2
CONFIG_PATH = "config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config


def get_PartD():
    PART_D_CONFIG = load_config()["model"]["PartD"]
    model = Decoder(
        input_token_size=PART_D_CONFIG["input_token_size"],
        output_channel_nums=PART_D_CONFIG["output_channel_nums"],
        output_img_size=PART_D_CONFIG["output_img_size"],
    )
    part_d = PartD(
        model=model,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss(),
        learning_rate=PART_D_CONFIG["learning_rate"],
    )
    return part_d

if __name__ == "__main__":
    tokens = torch.rand(TEST_BATCH_SIZE, 128)
    targets = torch.rand(TEST_BATCH_SIZE, 3, 256, 256)

    partd = get_PartD()

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = partd.train(tokens, targets)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

    outputs = partd(tokens)
    print("outputs.shape", outputs.shape)
