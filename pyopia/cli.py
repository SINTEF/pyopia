'''
PyOPIA top-level code primarily for managing cmd line entry points
'''

import typer
import toml
from glob import glob
import os
import datetime
import traceback
import logging
from rich.progress import track, Progress
import pandas as pd
import numpy as np
import threading

import pyopia.background
import pyopia.instrument.silcam
import pyopia.instrument.holo
import pyopia.instrument.common
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
def modify_config(existing_filename: str, modified_filename: str,
                  raw_files=None,
                  step_name=None, modify_arg=None, modify_value=None):
    '''Modify a existing config.toml file and write a new one to disc

    Parameters
    ----------
    existing_filename : str
        e.g. config.toml
    modified_filename : str
        e.g. config_new.toml
    raw_files : str, optional
        modify the raw file input in the `[general]` settings, by default None
    step_name : str, optional
        the name of the step to modify e.g. `segmentation`, by default None
    modify_arg : str, optional
        the name of the step to modify e.g. `threshold`.
        existing arguments will be overwritten, non-existent arguments will be created, by default None
    modify_value : str or floar, optional
        new value to attach to the 'modify_arg' setting e.g. 0.85.
        Accepts either string or float input, by default None
    '''
    toml_settings = pyopia.io.load_toml(existing_filename)

    if raw_files is not None:
        toml_settings['general']['raw_files'] = raw_files

    if step_name is not None:
        try:
            modify_value = float(modify_value)
        except ValueError:
            pass

        toml_settings['steps'][step_name][modify_arg] = modify_value

    with open(modified_filename, "w") as toml_file:
        toml.dump(toml_settings, toml_file)


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
def process(config_filename: str, chunks=1):
    '''Run a PyOPIA processing pipeline based on given a config.toml

    Parameters
    ----------
    config_filename : str
        config filename
    chunks : int, optional
        split the dataset into chucks, and process in parallell, by default 1

    '''
    chunks = int(chunks)
    from pyopia.io import load_toml
    from pyopia.pipeline import Pipeline

    with Progress(transient=True) as progress:
        progress.console.print("[blue]LOAD CONFIG")
        pipeline_config = load_toml(config_filename)

        setup_logging(pipeline_config)
        logger = logging.getLogger()

        progress.console.print("[blue]OBTAIN FILE LIST")
        files = sorted(glob(pipeline_config['general']['raw_files']))

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

        chunked_files, pipeline_config, initial_data = prepare_chunking(files, chunks, pipeline_config, Pipeline, logger)

        def process_file_list(file_list, inital_data, c):
            processing_pipeline = Pipeline(pipeline_config)
            if inital_data['imbg'] is not None:
                processing_pipeline.data['imbg'] = inital_data['imbg']
            for filename in track(file_list, description=f'[blue]Processing progress (thread {c})',
                                  disable=c != 0):
                try:
                    processing_pipeline.run(filename)
                except Exception as e:
                    progress.console.print("[red]An error occured in processing, " +
                                           "skipping rest of pipeline and moving to next image.")
                    logger.error(e)
                    logger.debug(''.join(traceback.format_tb(e.__traceback__)))

        for c, chunk in enumerate(chunked_files):
            job = threading.Thread(target=process_file_list, args=(chunk, initial_data, c, ))
            job.start()


@app.command()
def merge_mfdata(path_to_data: str, prefix='*'):
    '''Combine a multi-file directory of STATS.nc files into a single '-STATS.nc' file
    that can then be loaded with {func}`pyopia.io.load_stats`

    Parameters
    ----------
    path_to_data : str
        Folder name containing nc files with pattern '*Image-D*-STATS.nc'

    prefix : str
        Prefix to multi-file dataset (for replacing the wildcard in '*Image-D*-STATS.nc').
        Defaults to '*'
    '''
    pyopia.io.merge_and_save_mfdataset(path_to_data, prefix=prefix)


def setup_logging(pipeline_config):
    '''Configure logging

    Parameters
    ----------
    pipeline_config : dict
        TOML settings
    '''
    # Get user parameters or default values for logging
    log_file = pipeline_config['general'].get('log_file', None)
    log_level_name = pipeline_config['general'].get('log_level', 'INFO')
    log_level = getattr(logging, log_level_name)
    print(log_level_name, log_level, log_file)

    # Configure logger
    log_format = '%(asctime)s %(levelname)s [%(module)s.%(funcName)s] %(message)s'
    logging.basicConfig(level=log_level, format=log_format, filename=log_file,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger()
    logger.info(f'PyOPIA process started {pd.Timestamp.now()}')


def prepare_chunking(files, chunks, pipeline_config, Pipeline, logger):
    assert chunks > 0, 'you must have at least one chunk'

    initial_data = dict()
    initial_data['imbg'] = None
    chunk_mode = 'resample'

    if chunks > 1:
        pipeline_config['steps']['output']['append'] = False

        if 'correctbackground' in pipeline_config['steps']:
            background_pipeline = Pipeline(pipeline_config)
            if pipeline_config['steps']['correctbackground']['bgshift_function'] == 'pass':
                logger.info(f'Pre-calculating background from first {
                    pipeline_config['steps']['correctbackground']['average_window']} images')
                for file in files[0:pipeline_config['steps']['correctbackground']['average_window']]:
                    background_pipeline.run(file)

                pipeline_config['steps']['correctbackground']['average_window'] = 0
                logger.debug('average_window set to 0 to disable future background creation')

                initial_data['imbg'] = background_pipeline.data['imbg']
                chunk_mode = 'block'

    logger.info(f'Chunk mode: {chunk_mode}')
    chunked_files = chunk_files(files, chunks, mode=chunk_mode)
    return chunked_files, pipeline_config, initial_data


def chunk_files(files, chunks, mode='resample'):
    match mode:
        case 'block':
            n = int(np.ceil(len(files) / chunks))
            return [files[i:i + n] for i in range(0, len(files), n)]
        case 'resample':
            return [files[i:len(files):chunks] for i in range(0, chunks)]
    raise Exception("mode can either be 'block' or 'resample'")


if __name__ == "__main__":
    app()
