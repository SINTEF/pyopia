import os
from pathlib import Path
import pytest
import pandas as pd
from pyopia.io import write_stats, load_stats, get_cf_metadata_spec
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


if __name__ == "__main__":
    pytest.main()
