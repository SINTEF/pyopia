'''
A high level test for the basic processing pipeline from command line.

'''
import tempfile
import testdata
from glob import glob
import os
from typer.testing import CliRunner

from pyopia.cli import app

runner = CliRunner()


def test_app():
    print('pyopia process --help')
    result = runner.invoke(app, ["process", "--help"])
    assert result.exit_code == 0
    print(result.stdout)

    with tempfile.TemporaryDirectory() as tempdir:
        print('tmpdir created:', tempdir)
        print('download test data')
        infolder = testdata.get_folder_from_holo_repository("holo_test_data_01")
        print('download completed to ' + infolder + '. Reducing file list')
        files = sorted(glob(os.path.join(infolder, '*.pgm')))
        for f in files[6:]:
            os.remove(f)
        print('pyopia process')
        result = runner.invoke(app, ["process", "config-holo.toml"])
        assert result.exit_code == 0
        print(result.stdout)


if __name__ == "__main__":
    test_app()
