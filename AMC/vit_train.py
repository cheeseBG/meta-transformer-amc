import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
#from vit_pytorch import ViT
from models.vit import ViT
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
# vit_model = ViT(
#     image_size=1024,
#     patch_size=4,
#     num_classes=11,
#     dim=768,
#     depth=12,
#     heads=12,
#     mlp_dim=3072
# )
vit_model = ViT(
    in_channels=1,
    patch_size=4,
    num_classes=13,
    embed_dim=4,
    num_layers=2,
    num_heads=4,
    mlp_dim=2
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
        n_way = len(sample.keys())
        n_support = 5
        n_query = 10

        """
        support shape: [K_way, num_support, 1, I/Q, data_length]
        query shape: [K_way, num_query, 1, I/Q, data_length]
        """
        x_support = None
        x_query = None
        for label in sample.keys():
            if x_support is None:
                x_support = np.array([np.array(iq) for iq in sample[label]['support']])
            else:
                x_support = np.vstack([x_support, np.array([np.array(iq) for iq in sample[label]['support']])])
            if x_query is None:
                x_query = np.array([np.array(iq) for iq in sample[label]['query']])
            else:
                x_query = np.vstack([x_query, np.array([np.array(iq) for iq in sample[label]['query']])])

        x_support = torch.from_numpy(x_support)
        x_query = torch.from_numpy(x_query)


        # Extract the support set embeddings using the ViT model
        support_set_embeddings = vit_model(x_support)
        support_set_embeddings = support_set_embeddings.view(n_way, n_support, support_set_embeddings.size(-1))

        # Compute the prototype for each class
        prototypes = support_set_embeddings.mean(dim=1)

        # Extract the query set embeddings using the ViT model
        query_set_embeddings = vit_model(x_query)

        # Compute the distance between the query set and prototypes
        distances = torch.cdist(query_set_embeddings, prototypes.unsqueeze(0))
        print(distances.shape)
        exit()
        labels = np.array([[float(val)] for val in sample.keys() for i in range(10)])

        # Calculate the loss using the cross-entropy loss
        loss = F.cross_entropy(-distances, np.array(labels))

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