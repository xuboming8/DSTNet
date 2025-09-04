"""
Shared pytest fixtures for BasicSR testing.

This module provides common fixtures used across different test modules.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Create a temporary file in the temp directory."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    return temp_file_path


@pytest.fixture
def sample_image_data() -> np.ndarray:
    """Generate sample image data for testing."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_video_data() -> np.ndarray:
    """Generate sample video data for testing."""
    return np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        'name': 'test_model',
        'model_type': 'sr',
        'scale': 4,
        'num_gpu': 1,
        'manual_seed': 42,
        'datasets': {
            'train': {
                'name': 'test_train',
                'type': 'test_dataset',
                'dataroot_gt': '/fake/path/gt',
                'dataroot_lq': '/fake/path/lq',
            }
        },
        'network_g': {
            'type': 'test_arch',
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
        },
        'train': {
            'optim_g': {
                'type': 'Adam',
                'lr': 1e-4,
            },
            'scheduler': {
                'type': 'MultiStepLR',
                'milestones': [100000],
                'gamma': 0.5,
            },
            'total_iter': 200000,
            'warmup_iter': -1,
        },
        'val': {
            'val_freq': 5000,
            'save_img': True,
            'metrics': {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 4,
                    'test_y_channel': False,
                }
            }
        },
        'path': {
            'pretrain_network_g': None,
            'strict_load_g': True,
            'resume_state': None,
        }
    }


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    mock_device = Mock()
    mock_device.type = 'cpu'
    return mock_device


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def mock_model():
    """Provide a mock PyTorch model for testing."""
    model = Mock()
    model.eval = Mock()
    model.train = Mock()
    model.parameters = Mock(return_value=[])
    model.state_dict = Mock(return_value={})
    model.load_state_dict = Mock()
    model.forward = Mock()
    return model


@pytest.fixture
def mock_optimizer():
    """Provide a mock optimizer for testing."""
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    optimizer.state_dict = Mock(return_value={})
    optimizer.load_state_dict = Mock()
    return optimizer


@pytest.fixture
def mock_scheduler():
    """Provide a mock learning rate scheduler for testing."""
    scheduler = Mock()
    scheduler.step = Mock()
    scheduler.state_dict = Mock(return_value={})
    scheduler.load_state_dict = Mock()
    scheduler.get_last_lr = Mock(return_value=[1e-4])
    return scheduler


@pytest.fixture
def sample_dataset_structure(temp_dir: Path) -> Dict[str, Path]:
    """Create a sample dataset directory structure."""
    structure = {
        'gt_dir': temp_dir / 'gt',
        'lq_dir': temp_dir / 'lq', 
        'meta_file': temp_dir / 'meta_info.txt'
    }
    
    # Create directories
    structure['gt_dir'].mkdir(parents=True, exist_ok=True)
    structure['lq_dir'].mkdir(parents=True, exist_ok=True)
    
    # Create some sample files
    for i in range(3):
        (structure['gt_dir'] / f'img_{i:03d}.png').write_text(f'gt_image_{i}')
        (structure['lq_dir'] / f'img_{i:03d}.png').write_text(f'lq_image_{i}')
    
    # Create meta info file
    meta_content = '\n'.join([f'img_{i:03d}.png' for i in range(3)])
    structure['meta_file'].write_text(meta_content)
    
    return structure


@pytest.fixture
def sample_checkpoint_data() -> Dict[str, Any]:
    """Provide sample checkpoint data for testing."""
    return {
        'model': {'param1': [1, 2, 3]},
        'optimizer': {'lr': 1e-4},
        'scheduler': {'step': 1000},
        'epoch': 10,
        'iter': 50000,
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables and settings."""
    # Set test-specific environment variables
    monkeypatch.setenv('BASICSR_EXT', 'False')  # Disable CUDA extensions for tests
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')  # Disable CUDA for tests
    
    # Mock torch.cuda.is_available to return False for tests
    try:
        import torch
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    except ImportError:
        pass  # torch not available, skip mocking


@pytest.fixture
def mock_file_client():
    """Mock file client for testing."""
    client = Mock()
    client.get = Mock()
    client.put = Mock()
    client.exists = Mock(return_value=True)
    client.isdir = Mock(return_value=False)
    client.isfile = Mock(return_value=True)
    client.listdir = Mock(return_value=[])
    return client