from pathlib import Path

import pytest


@pytest.fixture
def visual_output_path() -> Path:
    """Per-test folder rooted under build/test-images/."""
    test_images_path = Path("build/test-images")
    test_images_path.mkdir(parents=True, exist_ok=True)
    return test_images_path
