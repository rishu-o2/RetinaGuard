"""Unit tests for FundusCameraAugmentor and camera-simulating transformations."""

import numpy as np
import pytest

from retinaguard_ml.data.augmentations import FundusCameraAugmentor


@pytest.fixture
def synthetic_retina():
    """Create a synthetic 256x256 RGB retina-like uint8 image."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Red-orange fundus disc
    y, x = np.ogrid[:256, :256]
    mask = (x - 128) ** 2 + (y - 128) ** 2 <= 110 ** 2
    img[mask] = [180, 80, 40]
    # Vessel-like texture
    img[120:136, 120:136] = [220, 160, 60]  # Optic disc
    img[80:85, 100:150] = [120, 20, 15]     # Vessel/lesion
    return img


def test_augmentor_output_shape_and_dtype(synthetic_retina):
    aug = FundusCameraAugmentor(seed=42)
    out = aug(synthetic_retina)
    assert out.shape == synthetic_retina.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_brightness_contrast(synthetic_retina):
    aug = FundusCameraAugmentor(
        p_brightness_contrast=1.0,
        contrast_range=(1.2, 1.2),
        brightness_range=(20.0, 20.0),
        p_hflip=0.0,
        p_vflip=0.0,
        p_rotate=0.0,
        p_gamma=0.0,
        p_hue_saturation=0.0,
        p_white_balance=0.0,
        p_blur=0.0,
        p_sharpness=0.0,
        p_jpeg=0.0,
        p_vignetting=0.0,
        seed=1,
    )
    out = aug(synthetic_retina)
    assert not np.array_equal(out, synthetic_retina)
    # Brightness + contrast should increase mean pixel value
    assert out.mean() > synthetic_retina.mean()


def test_gamma_transformation(synthetic_retina):
    aug = FundusCameraAugmentor(
        p_gamma=1.0,
        gamma_range=(0.6, 0.6),
        p_hflip=0.0,
        p_vflip=0.0,
        p_rotate=0.0,
        p_brightness_contrast=0.0,
        p_hue_saturation=0.0,
        p_white_balance=0.0,
        p_blur=0.0,
        p_sharpness=0.0,
        p_jpeg=0.0,
        p_vignetting=0.0,
        seed=2,
    )
    out = aug(synthetic_retina)
    assert not np.array_equal(out, synthetic_retina)
    assert out.dtype == np.uint8


def test_hue_saturation(synthetic_retina):
    aug = FundusCameraAugmentor(
        p_hue_saturation=1.0,
        hue_shift_limit=6,
        sat_shift_limit=15,
        p_hflip=0.0,
        p_vflip=0.0,
        p_rotate=0.0,
        p_brightness_contrast=0.0,
        p_gamma=0.0,
        p_white_balance=0.0,
        p_blur=0.0,
        p_sharpness=0.0,
        p_jpeg=0.0,
        p_vignetting=0.0,
        seed=3,
    )
    out = aug(synthetic_retina)
    assert not np.array_equal(out, synthetic_retina)
    assert out.shape == synthetic_retina.shape


def test_white_balance(synthetic_retina):
    aug = FundusCameraAugmentor(
        p_white_balance=1.0,
        wb_temp_range=(10.0, 10.0),
        p_hflip=0.0,
        p_vflip=0.0,
        p_rotate=0.0,
        p_brightness_contrast=0.0,
        p_gamma=0.0,
        p_hue_saturation=0.0,
        p_blur=0.0,
        p_sharpness=0.0,
        p_jpeg=0.0,
        p_vignetting=0.0,
        seed=4,
    )
    out = aug(synthetic_retina)
    # Warm shift boosts red channel
    assert out[:, :, 0].mean() >= synthetic_retina[:, :, 0].mean()


def test_blur_and_sharpness(synthetic_retina):
    aug_blur = FundusCameraAugmentor(
        p_blur=1.0,
        blur_kernel_choices=(5,),
        p_hflip=0.0,
        p_vflip=0.0,
        p_rotate=0.0,
        p_brightness_contrast=0.0,
        p_gamma=0.0,
        p_hue_saturation=0.0,
        p_white_balance=0.0,
        p_sharpness=0.0,
        p_jpeg=0.0,
        p_vignetting=0.0,
        seed=5,
    )
    blurred = aug_blur(synthetic_retina)
    assert not np.array_equal(blurred, synthetic_retina)

    aug_sharp = FundusCameraAugmentor(
        p_sharpness=1.0,
        sharpness_strength_range=(0.7, 0.7),
        p_hflip=0.0,
        p_vflip=0.0,
        p_rotate=0.0,
        p_brightness_contrast=0.0,
        p_gamma=0.0,
        p_hue_saturation=0.0,
        p_white_balance=0.0,
        p_blur=0.0,
        p_jpeg=0.0,
        p_vignetting=0.0,
        seed=6,
    )
    sharpened = aug_sharp(synthetic_retina)
    assert not np.array_equal(sharpened, synthetic_retina)


def test_jpeg_compression(synthetic_retina):
    aug = FundusCameraAugmentor(
        p_jpeg=1.0,
        jpeg_quality_range=(50, 50),
        p_hflip=0.0,
        p_vflip=0.0,
        p_rotate=0.0,
        p_brightness_contrast=0.0,
        p_gamma=0.0,
        p_hue_saturation=0.0,
        p_white_balance=0.0,
        p_blur=0.0,
        p_sharpness=0.0,
        p_vignetting=0.0,
        seed=7,
    )
    out = aug(synthetic_retina)
    assert not np.array_equal(out, synthetic_retina)
    assert out.shape == synthetic_retina.shape


def test_vignetting(synthetic_retina):
    aug = FundusCameraAugmentor(
        p_vignetting=1.0,
        vignette_falloff_range=(0.35, 0.35),
        p_hflip=0.0,
        p_vflip=0.0,
        p_rotate=0.0,
        p_brightness_contrast=0.0,
        p_gamma=0.0,
        p_hue_saturation=0.0,
        p_white_balance=0.0,
        p_blur=0.0,
        p_sharpness=0.0,
        p_jpeg=0.0,
        seed=8,
    )
    out = aug(synthetic_retina)
    assert not np.array_equal(out, synthetic_retina)
    # Peripheral pixels should be darker
    assert out[5, 5].mean() <= synthetic_retina[5, 5].mean()


def test_geometric_flips_and_rotations(synthetic_retina):
    aug = FundusCameraAugmentor(
        p_hflip=1.0,
        p_vflip=0.0,
        p_rotate=0.0,
        p_brightness_contrast=0.0,
        p_gamma=0.0,
        p_hue_saturation=0.0,
        p_white_balance=0.0,
        p_blur=0.0,
        p_sharpness=0.0,
        p_jpeg=0.0,
        p_vignetting=0.0,
        seed=9,
    )
    flipped = aug(synthetic_retina)
    assert np.array_equal(flipped, np.fliplr(synthetic_retina))

