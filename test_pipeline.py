import pytest
import torch
from train import SimpleCNN
import train as trn

@pytest.fixture
def model():
    """Fixture to provide the model instance."""
    return SimpleCNN()

def test_total_parameters(model):
    """Test that the model has fewer than 100,000 parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, exceeding the limit."

def test_input_compatibility(model):
    """Test that the model accepts 28x28 input without errors."""
    sample_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(sample_input)
    except Exception as e:
        pytest.fail(f"Model failed with 28x28 input: {e}")
    
def test_output_shape(model):
    """Test that the model output has 10 units."""
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    assert output.shape[1] == 10, f"Model output shape mismatch: {output.shape}"

def test_accuracy_greater_than_95():
    accuracy = trn.train()
    assert accuracy >= 95, f"Model has {accuracy} accuracy, not meeting the requirement."

def test_gradient_flow(model):
    """Test that all model parameters receive gradients during backpropagation."""
    sample_input = torch.randn(1, 1, 28, 28)
    sample_target = torch.tensor([1])  # Assuming a single target class
    criterion = torch.nn.CrossEntropyLoss()

    # Perform a forward pass
    output = model(sample_input)
    loss = criterion(output, sample_target)

    # Backward pass
    loss.backward()

    # Check if all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient not computed for parameter: {name}"

@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_batch_compatibility(model, batch_size):
    """Test that the model works with varying batch sizes."""
    sample_input = torch.randn(batch_size, 1, 28, 28)
    try:
        output = model(sample_input)
        assert output.shape[0] == batch_size, f"Model output batch size mismatch: expected {batch_size}, got {output.shape[0]}"
    except Exception as e:
        pytest.fail(f"Model failed with batch size {batch_size}: {e}")

def test_output_probabilities(model):
    """Test that the model output is a valid probability distribution."""
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Check that the probabilities sum to 1
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5), "Output probabilities do not sum to 1."

    # Ensure no probabilities are negative
    assert (probabilities >= 0).all(), "Output contains negative probabilities."

# Run tests only if the script is executed directly (optional with pytest)
if __name__ == "__main__":
    pytest.main()
