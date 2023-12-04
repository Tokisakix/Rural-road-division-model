import json
import torch
import torch.nn as nn
import torch.optim as optim

class CAMGenerator(nn.Module):
    '''
    A decoder module for generating Class Activation Maps (CAM).

    Args:
    - input_size (int): The size of the input features.
    - output_size (int): The size of the output features.

    Returns:
    - cam_output: The output features from the decoder.
    '''
    def __init__(self, input_size, output_size):
        super(CAMGenerator, self).__init__()
        self.decoder = nn.Linear(input_size, output_size)

    def forward(self, x):
        cam_output = self.decoder(x)
        return cam_output

class CAMGeneratorNetwork:
    '''
    A training network for the CAMGenerator.

    Args:
    - vit_model: A pretrained vision transformer model.
    - classifier: A pretrained classifier.
    - input_data: The input data for the network.
    - ground_truth_labels: The ground truth labels for the input data.
    '''
    def __init__(self, vit_model, classifier, input_data, ground_truth_labels):
        self.vit_model = vit_model
        self.classifier = classifier
        self.input_data = input_data
        self.ground_truth_labels = ground_truth_labels

        input_size = self.vit_model.token_embeddings.size(-1)  # Size of token embeddings
        output_size = 128  # Size of CAM output
        self.cam_generator = CAMGenerator(input_size, output_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.cam_generator.parameters(), lr=0.001)

    def boundary_loss(self, cam_output):
        '''
        Calculate and return the boundary loss for the CAM output.
        '''
        pass

    def train(self, num_epochs=10):
        '''
        Train the CAMGeneratorNetwork for a specified number of epochs.
        '''
        all_cam_outputs = []
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            token_embeddings = self.vit_model(self.input_data)

            cam_output = self.cam_generator(token_embeddings)

            logits = self.classifier(cam_output)

            b_loss = self.boundary_loss(cam_output)
            c_loss = self.criterion(logits, self.ground_truth_labels)
            total_loss = b_loss + c_loss

            total_loss.backward()
            self.optimizer.step()

            all_cam_outputs.append(cam_output.detach())

            print(f"Epoch [{epoch + 1}/{num_epochs}], Boundary Loss: {b_loss.item()}, Classification Loss: {c_loss.item()}")

        final_cam_output = torch.cat(all_cam_outputs, dim=0)

        return final_cam_output

# Example usage:
# pretrained_classifier = ...
# cam_network = CAMGeneratorNetwork(vit_model, pretrained_classifier, input_data, ground_truth_labels)
# final_cam_output = cam_network.train(num_epochs=10)

class Decoder(nn.Module):
    def __init__(self, input_token_size, output_channel_nums, output_img_size):
        super(Decoder, self).__init__()
        self.func = nn.Linear(128, 3 * 256 * 256)
        self.input_token_size = input_token_size
        self.output_channel_nums = output_channel_nums
        self.output_img_size = output_img_size
        return
    
    def forward(self, tokens):
        out = self.func(tokens)
        out = out.reshape(tokens.shape[0], self.output_channel_nums, self.output_img_size, self.output_img_size)
        return out

class PartD(nn.Module):
    def __init__(self, model, optimizer, criterion, learning_rate):
        super(PartD, self).__init__()
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr = learning_rate)
        self.criterion = criterion
        return
    
    def train(self, inputs, targets):
        outputs = self.model(inputs)
        self.optimizer.zero_grad()
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()
    
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
        input_token_size = PART_D_CONFIG["input_token_size"],
        output_channel_nums = PART_D_CONFIG["output_channel_nums"],
        output_img_size = PART_D_CONFIG["output_img_size"],
    )
    part_d = PartD(
        model = model,
        optimizer = torch.optim.Adam,
        criterion = torch.nn.MSELoss(),
        learning_rate = PART_D_CONFIG["learning_rate"],
    )
    return part_d

if __name__ == "__main__":
    tokens      = torch.rand(TEST_BATCH_SIZE * 16, 128)
    targets     = torch.rand(TEST_BATCH_SIZE * 16, 3, 256, 256)
    
    partd = get_PartD()
    outputs = partd(tokens)
    partd.train(tokens, targets)
    print("outputs.shape", outputs.shape)