import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from vit_pytorch import ViT
import numpy as np
from data.dataset import FewShotDataset
import torch.utils.data as DATA
from runner.utils import get_config

config = get_config('config.yaml')
train_data = FewShotDataset(config["dataset_path"],
                            num_support=config["num_support"],
                            num_query=config["num_query"],
                            robust=True,
                            snr_range=config["snr_range"])

train_dataloader = DATA.DataLoader(train_data, batch_size=1, shuffle=True)

# Initialize the ViT model as the base encoder
vit_model = ViT(
    image_size=1024,
    patch_size=4,
    num_classes=11,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072
)

# Set the optimizer and the learning rate scheduler
optimizer = torch.optim.Adam(vit_model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Set the number of epochs
num_epochs = 50

# Train the model
for epoch in range(num_epochs):
    vit_model.train()
    epoch_loss = 0

    for sample in train_dataloader:

        for label in sample.keys():
            # Create a random support set
            support_set = np.array(sample[label]['support'])

            # Extract the support set embeddings using the ViT model
            support_set_embeddings = vit_model(support_set)

            # Compute the prototype for each class
            prototypes = support_set_embeddings.mean(dim=0)

            # Extract the query set embeddings using the ViT model
            query_set_embeddings = vit_model(np.array(sample[label]['query']))

            # Compute the distance between the query set and prototypes
            distances = torch.cdist(query_set_embeddings, prototypes.unsqueeze(0))

            # Calculate the loss using the cross-entropy loss
            loss = F.cross_entropy(-distances, label)

            # Update the model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    # Update the learning rate
    lr_scheduler.step()

    # Print the epoch loss
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss / len(train_dataloader)))

# Save the model weights
#torch.save(vit_model.state_dict(), "vit_model_weights.pth")