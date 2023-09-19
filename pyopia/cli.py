'''
PyOPIA top-level code primarily for managing cmd line entry points
'''

import typer
import toml
from glob import glob
import pyopia.background
import pyopia.classify
import pyopia.instrument.silcam
import pyopia.instrument.holo
import pyopia.io
import pyopia.pipeline
import pyopia.plotting
import pyopia.process
import pyopia.statistics

app = typer.Typer()


@app.command()
def docs():
    '''Open browser at PyOPIA's readthedocs page
    '''
    print("Opening PyOPIA's docs")
    typer.launch("https://pyopia.readthedocs.io")


@app.command()
def generate_config(instrument: str, raw_files: str, model_path: str, outfolder: str, output_prefix: str):
    '''Put an example config.toml file in the current directory

    Parameters
    ----------
    instrument : str
        either <silcam> or <holo>
    raw_files : str
        raw_files
    model_path : str
        model_path
    outfolder : str
        outfolder
    output_prefix : str
        output_prefix
    '''
    match instrument:
        case 'silcam':
            pipeline_config = pyopia.instrument.silcam.generate_config(raw_files, model_path, outfolder, output_prefix)
        case 'holo':
            pipeline_config = pyopia.instrument.holo.generate_config(raw_files, model_path, outfolder, output_prefix)

    config_filename = instrument + "-config.toml"
    with open(config_filename, "w") as toml_file:
        toml.dump(pipeline_config, toml_file)


@app.command()
def process(config_filename: str, ):
    '''Run a PyOPIA processing pipeline based on given a config.toml
    '''
    from pyopia.io import load_toml
    from pyopia.pipeline import Pipeline

    pipeline_config = load_toml(config_filename)

    files = sorted(glob(pipeline_config['general']['raw_files']))

    processing_pipeline = Pipeline(pipeline_config)

    for filename in files:
        processing_pipeline.run(filename)


if __name__ == "__main__":
    app()
