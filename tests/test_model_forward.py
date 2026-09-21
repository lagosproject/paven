import pytest
import torch
from paven.model import VideoSaliencyModel, PavenModel


def test_model_forward_synthetic():
    """Verify that VideoSaliencyModel performs a valid forward pass on synthetic clips."""
    model = VideoSaliencyModel(
        transformer_in_channel=32,
        nhead=4,
        use_upsample=True,
        num_hier=3,
        num_clips=32
    )
    model.eval()

    # Synthetic clip tensor: (Batch=1, Channels=3, Time=32, Height=224, Width=384)
    dummy_input = torch.randn(1, 3, 32, 224, 384, dtype=torch.float32)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (1, 224, 384)
    assert output.shape == (1, 224, 384)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_paven_model_wrapper():
    """Verify PavenModel wrapper initialization."""
    wrapper = PavenModel(num_clips=32)
    dummy_input = torch.randn(1, 3, 32, 224, 384)
    with torch.no_grad():
        out = wrapper(dummy_input)
    assert out.shape == (1, 224, 384)
