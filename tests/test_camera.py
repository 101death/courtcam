import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import camera

def test_validate_resolution_valid():
    assert camera.validate_resolution(640, 480) == (640, 480)
    assert camera.validate_resolution("1024", "768") == (1024, 768)


def test_validate_resolution_invalid_defaults_to_hd():
    assert camera.validate_resolution("abc", "def") == camera.DEFAULT_RESOLUTION
    assert camera.validate_resolution(None, None) == camera.DEFAULT_RESOLUTION


def test_validate_resolution_below_minimum():
    assert camera.validate_resolution(50, 50) == (160, 120)
    assert camera.validate_resolution(159, 130) == (160, 120)
    assert camera.validate_resolution(200, 100) == (160, 120)

