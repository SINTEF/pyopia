import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from pyopia.io import write_stats, load_stats, get_cf_metadata_spec, ImageToDisc
from pyopia.instrument.silcam import generate_config


pipeline_config = generate_config(
    raw_files="images/*.silc",
    model_path="None",
    outfolder="processed",
    output_prefix="test",
)


def test_write_and_load_stats(tmp_path: Path):
    # Create a temporary directory for testing
    temp_dir = tmp_path / "test_data"
    temp_dir.mkdir()

    # Create a sample DataFrame to write
    data = {
        "major_axis_length": [10.5, 20.3],
        "minor_axis_length": [5.2, 10.1],
        "equivalent_diameter": [7.8, 15.2],
        "saturation": [50.0, 75.0],
    }
    # Convert timestamp to datetime format
    stats_df = pd.DataFrame(data)
    stats_df["timestamp"] = pd.to_datetime(
        ["2025-04-25T10:00:00", "2025-04-25T10:05:00"]
    )

    # Define the output file path
    output_file = os.path.join(temp_dir, "test")

    # Provide a minimal valid settings object
    config = pipeline_config

    # Write the stats to a NetCDF file
    write_stats(stats_df, output_file, settings=config, dataformat="nc", append=True)

    # Check if the file was created
    assert os.path.exists(output_file + "-STATS.nc")

    # Load the stats back
    loaded_stats = load_stats(output_file + "-STATS.nc")

    # Verify the data matches
    for var in data.keys():
        assert var in loaded_stats.data_vars
        assert all(loaded_stats[var].values == stats_df[var].values)

    assert "timestamp" in loaded_stats.coords

    # Verify CF_METADATA attributes
    for var, metadata in get_cf_metadata_spec().items():
        if var in loaded_stats.data_vars:
            for attr, value in metadata.items():
                assert loaded_stats[var].attrs.get(attr) == value

    # Check that version tag is in the attributes
    assert "PyOPIA_version" in loaded_stats.attrs


def test_image_to_disc_separate(tmp_path: Path):
    """Test ImageToDisc saves separate images for each pipeline key."""
    output_folder = str(tmp_path / "output_images")

    saver = ImageToDisc(
        output_folder=output_folder,
        image_keys=['imraw', 'im_corrected', 'imbw'],
        scale_factor=1.0,
        collage=False,
    )

    # Create fake pipeline data with a mix of image types
    data = {
        'filename': '/fake/path/test_image.silc',
        'imraw': np.random.rand(100, 120, 3),
        'im_corrected': np.random.rand(100, 120, 3),
        'imbw': np.random.rand(100, 120) > 0.5,  # boolean segmentation mask
    }

    result = saver(data)

    # Check data is returned unmodified
    assert result is data

    # Check output files exist
    assert os.path.isfile(os.path.join(output_folder, 'test_image_imraw.png'))
    assert os.path.isfile(os.path.join(output_folder, 'test_image_im_corrected.png'))
    assert os.path.isfile(os.path.join(output_folder, 'test_image_imbw.png'))


def test_image_to_disc_collage(tmp_path: Path):
    """Test ImageToDisc saves a single collage image."""
    output_folder = str(tmp_path / "output_collage")

    saver = ImageToDisc(
        output_folder=output_folder,
        image_keys=['imraw', 'im_corrected'],
        collage=True,
    )

    data = {
        'filename': '/fake/path/sample.silc',
        'imraw': np.random.rand(80, 100, 3),
        'im_corrected': np.random.rand(80, 100, 3),
    }

    result = saver(data)

    assert result is data
    assert os.path.isfile(os.path.join(output_folder, 'sample_collage.png'))


def test_image_to_disc_scale_factor(tmp_path: Path):
    """Test ImageToDisc applies scale factor when saving separate images."""
    output_folder = str(tmp_path / "output_scaled")

    saver = ImageToDisc(
        output_folder=output_folder,
        image_keys=['imraw'],
        scale_factor=0.5,
        collage=False,
    )

    data = {
        'filename': '/fake/path/scaled_test.silc',
        'imraw': np.random.rand(100, 120, 3),
    }

    saver(data)

    out_file = os.path.join(output_folder, 'scaled_test_imraw.png')
    assert os.path.isfile(out_file)

    # Load the saved image and verify it was scaled down
    import matplotlib.pyplot as plt
    saved_img = plt.imread(out_file)
    assert saved_img.shape[0] == 50
    assert saved_img.shape[1] == 60


def test_image_to_disc_missing_keys(tmp_path: Path):
    """Test ImageToDisc gracefully skips missing keys."""
    output_folder = str(tmp_path / "output_missing")

    saver = ImageToDisc(
        output_folder=output_folder,
        image_keys=['imraw', 'imbg', 'nonexistent_key'],
    )

    data = {
        'filename': '/fake/path/missing_test.silc',
        'imraw': np.random.rand(50, 60, 3),
        # 'imbg' and 'nonexistent_key' intentionally missing
    }

    result = saver(data)

    assert result is data
    # Only imraw should be saved
    assert os.path.isfile(os.path.join(output_folder, 'missing_test_imraw.png'))
    assert not os.path.isfile(os.path.join(output_folder, 'missing_test_imbg.png'))
    assert not os.path.isfile(os.path.join(output_folder, 'missing_test_nonexistent_key.png'))


def test_image_to_disc_2d_grayscale(tmp_path: Path):
    """Test ImageToDisc handles 2D grayscale images."""
    output_folder = str(tmp_path / "output_gray")

    saver = ImageToDisc(
        output_folder=output_folder,
        image_keys=['im_corrected'],
        scale_factor=0.5,
    )

    data = {
        'filename': '/fake/path/gray_test.png',
        'im_corrected': np.random.rand(100, 120),
    }

    saver(data)

    assert os.path.isfile(os.path.join(output_folder, 'gray_test_im_corrected.png'))


def test_image_to_disc_collage_mixed_types(tmp_path: Path):
    """Test collage with a mix of 2D (binary) and 3D (RGB) images."""
    output_folder = str(tmp_path / "output_collage_mixed")

    saver = ImageToDisc(
        output_folder=output_folder,
        image_keys=['imraw', 'imbw'],
        collage=True,
        scale_factor=0.5,
    )

    data = {
        'filename': '/fake/path/mixed_test.silc',
        'imraw': np.random.rand(80, 100, 3),
        'imbw': np.random.rand(80, 100) > 0.5,
    }

    saver(data)

    assert os.path.isfile(os.path.join(output_folder, 'mixed_test_collage.png'))


if __name__ == "__main__":
    pytest.main()
