import tempfile

import pytest
import torch
import torch.nn as nn

from torchdendrite.models.modules import DendriticLinear


def test_dendritic_linear_learnable_parameters():
    # Create a temporary directory for the model and optimizer checkpoints
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = tmp_dir + "/model_checkpoints"
        optimizer_dir = tmp_dir + "/optimizer_checkpoints"

        # Create a DendriticLinear layer
        model = DendriticLinear(in_features=10, out_features=5, resolution=5)

        # Define an optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Define a loss function
        loss_fn = torch.nn.MSELoss()

        # Create input and target tensors
        input_data = torch.rand(32, 10)
        target_data = torch.rand(32, 5)

        # Move tensors to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        # Forward pass
        output_data = model(input_data)

        # Compute loss
        loss = loss_fn(output_data, target_data)

        # Initialize the model and optimizer states
        model.train()
        optimizer.zero_grad()

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Save the model and optimizer states
        torch.save(model.state_dict(), model_dir + "/model_checkpoint.pt")
        torch.save(optimizer.state_dict(), optimizer_dir + "/optimizer_checkpoint.pt")

        # Load the model and optimizer states
        loaded_model = DendriticLinear(in_features=10, out_features=5, resolution=5)
        loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.01)

        loaded_model.load_state_dict(
            torch.load(model_dir + "/model_checkpoint.pt"), strict=True
        )
        loaded_optimizer.load_state_dict(
            torch.load(optimizer_dir + "/optimizer_checkpoint.pt")
        )

        # Check if the learnable parameters have been updated
        for name, param in loaded_model.named_parameters():
            if "buffer" not in name:
                assert (param.grad != 0).any(), f"Parameter {name} has zero gradients."


def test_dendritic_linear_integration_in_nn():
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.dendritic_linear = DendriticLinear(
                in_features=10, out_features=5, resolution=5
            )
            self.linear = nn.Linear(5, 1)

        def forward(self, x):
            x = self.dendritic_linear(x)
            x = self.linear(x)
            return x

    # Create a simple input tensor
    input_tensor = torch.rand(2, 10)

    # Instantiate the TestModel
    model = TestModel()

    # Forward pass through the model
    output = model(input_tensor)

    # Check the output shape
    assert output.shape == (2, 1)


@pytest.fixture()
def batch_size():
    for batch in [1, 5, 10]:
        yield batch


def test_dendritic_linear_batch_size(batch_size):
    # Create a DendriticLinear layer
    model = DendriticLinear(in_features=10, out_features=5, resolution=5)

    # Test with different batch sizes

    x = torch.rand(batch_size, 10)
    y = model(x)
    assert y.shape == (batch_size, 5)


def test_dendritic_linear_varying_input_dims():
    # Test case 1: 2D input tensor
    x_2d = torch.rand(5, 10)
    model_2d = DendriticLinear(in_features=10, out_features=5, resolution=5)
    y_2d = model_2d(x_2d)
    assert y_2d.shape == (5, 5)

    # Test case 2: 3D input tensor
    x_3d = torch.rand(3, 5, 10)
    model_3d = DendriticLinear(in_features=10, out_features=5, resolution=5)
    y_3d = model_3d(x_3d)
    assert y_3d.shape == (3, 5, 5)
