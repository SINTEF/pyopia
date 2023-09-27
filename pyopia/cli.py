'''
PyOPIA top-level code primarily for managing cmd line entry points
'''

import typer
import toml
from glob import glob
import os
import datetime
from rich.progress import track, Progress

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
        either `silcam` or `holo`
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

    with Progress(transient=True) as progress:
        progress.console.print("[blue]LOAD CONFIG")
        pipeline_config = load_toml(config_filename)

        progress.console.print("[blue]OBTAIN FILE LIST")
        files = sorted(glob(pipeline_config['general']['raw_files']))
        nfiles = len(files)

        progress.console.print('[blue]PREPARE FOLDERS')
        if 'output' not in pipeline_config['steps']:
            raise Exception('The given config file is missing an "output" step.\n' +
                            'This is needed to setup how to save data to disc.')
        output_datafile = pipeline_config['steps']['output']['output_datafile']
        os.makedirs(os.path.split(output_datafile)[:-1][0],
                    exist_ok=True)

        if os.path.isfile(output_datafile + '-STATS.nc'):
            dt_now = datetime.datetime.now().strftime('D%Y%m%dT%H%M%S')
            newname = output_datafile + '-conflict-' + str(dt_now) + '-STATS.nc'
            progress.console.print('[red]Renaming conflicting file to: ' +
                                   newname)
            os.rename(output_datafile + '-STATS.nc', newname)

        progress.console.print("[blue]INITIALISE PIPELINE")
        processing_pipeline = Pipeline(pipeline_config)

    for filename in track(files, description=f'[blue]Processing progress through {nfiles} files:'):
        processing_pipeline.run(filename)


if __name__ == "__main__":
    app()
