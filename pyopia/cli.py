'''
PyOPIA top-level code primarily for managing cmd line entry points
'''

import typer
import toml
import os
import datetime
import traceback
import logging
from rich.progress import track, Progress
import rich.progress
from rich.logging import RichHandler
import pandas as pd
import threading

import pyopia.background
import pyopia.instrument.silcam
import pyopia.instrument.holo
import pyopia.instrument.uvp
import pyopia.instrument.common
import pyopia.io
import pyopia.pipeline
import pyopia.plotting
import pyopia.process
import pyopia.statistics

app = typer.Typer()


def get_custom_progress_bar(description, disable):
    ''' Create a custom rich.progress.Progress object for displaying progress bars'''
    progress = Progress(
        rich.progress.TextColumn(description),
        rich.progress.BarColumn(),
        rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TextColumn("•"),
        rich.progress.TimeElapsedColumn(),
        rich.progress.TextColumn("•"),
        rich.progress.TimeRemainingColumn(),
        disable=disable
    )
    return progress


@app.command()
def docs():
    '''Open browser at PyOPIA's readthedocs page
    '''
    print("Opening PyOPIA's docs")
    typer.launch("https://pyopia.readthedocs.io")


@app.command()
def modify_config(existing_filename: str, modified_filename: str,
                  raw_files=None, pixel_size=None,
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
    pixel_size : str, optional
        modify the pixel size in the `[general]` settings, by default None
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
        toml_settings['general']['raw_files'] = f'{raw_files}'
    if pixel_size is not None:
        toml_settings['general']['pixel_size'] = float(pixel_size)

    if step_name is not None:
        try:
            if modify_arg == 'average_window':
                modify_value = int(modify_value)
            elif modify_arg == 'threshold':
                modify_value = float(modify_value)
            else:
                modify_value = str(modify_value)
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
        either `silcam`, `holo` or `uvp`
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
        case 'uvp':
            pipeline_config = pyopia.instrument.uvp.generate_config(raw_files, model_path, outfolder, output_prefix)

    config_filename = instrument + "-config.toml"
    with open(config_filename, "w") as toml_file:
        toml.dump(pipeline_config, toml_file)


@app.command()
def process(config_filename: str, num_chunks: int = 1):
    '''Run a PyOPIA processing pipeline based on given a config.toml

    Parameters
    ----------
    config_filename : str
        config filename
    numchunks : int, optional
        split the dataset into chucks, and process in parallell, by default 1

    '''
    from pyopia.io import load_toml
    from pyopia.pipeline import Pipeline

    with Progress(transient=True) as progress:
        progress.console.print("[blue]LOAD CONFIG")
        pipeline_config = load_toml(config_filename)

        setup_logging(pipeline_config)
        logger = logging.getLogger('rich')
        logger.info(f'PyOPIA process started {pd.Timestamp.now()}')

        check_chunks(num_chunks, pipeline_config)

        progress.console.print("[blue]OBTAIN IMAGE LIST")
        raw_files = pyopia.pipeline.FilesToProcess(pipeline_config['general']['raw_files'])
        progress.console.print(f"[blue]FOUND {len(raw_files)} IMAGES")
        conf_corrbg = pipeline_config['steps'].get('correctbackground', dict())
        average_window = conf_corrbg.get('average_window', 0)
        bgshift_function = conf_corrbg.get('bgshift_function', 'pass')
        raw_files.prepare_chunking(num_chunks, average_window, bgshift_function)

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
            logger.warning(f'Renaming conflicting file to: {newname}')
            os.rename(output_datafile + '-STATS.nc', newname)

        progress.console.print("[blue]INITIALISE PIPELINE")

    def process_file_list(file_list, c):
        processing_pipeline = Pipeline(pipeline_config)
        with get_custom_progress_bar(f'[blue]Processing progress (chunk {c})', disable=c!=1) as pbar:
            for filename in pbar.track(file_list, description=f'[blue]Processing progress (chunk {c})'):
                try:
                    logger.debug(f'Chunk {c} starting to process {filename}')
                    processing_pipeline.run(filename)
                except Exception as e:
                    logger.warning('[red]An error occured in processing, ' +
                                'skipping rest of pipeline and moving to next image.' +
                                f'(chunk {c})')
                    logger.error(e)
                    logger.debug(''.join(traceback.format_tb(e.__traceback__)))

    # With one chunk we keep the non-threaded functionality to ensure backwards compatibility
    if num_chunks == 1:
        process_file_list(raw_files, 0)
    else:
        for c, chunk in enumerate(raw_files.chunked_files):
            job = threading.Thread(target=process_file_list, args=(chunk, c, ))
            job.start()


@app.command()
def merge_mfdata(path_to_data: str, prefix='*', overwrite_existing_partials: bool = True,
                 chunk_size: int = None):
    '''Combine a multi-file directory of STATS.nc files into a single '-STATS.nc' file
    that can then be loaded with {func}`pyopia.io.load_stats`

    Parameters
    ----------
    path_to_data : str
        Folder name containing nc files with pattern '*Image-D*-STATS.nc'

    prefix : str
        Prefix to multi-file dataset (for replacing the wildcard in '*Image-D*-STATS.nc').
        Defaults to '*'

    overwrite_existing_partials : bool
        Do not reprocess existing merged netcdf files for each chunk if False.
        Otherwise reprocess (load) and overwrite. This can be used to restart
        or continue a previous merge operation as new files become available.

    chunk_size : int
        Process this many files together and store as partially merged netcdf files, which
        are then merged at the end. Default: None, process all files together.
    '''
    setup_logging({'general': {}})

    pyopia.io.merge_and_save_mfdataset(path_to_data, prefix=prefix,
                                       overwrite_existing_partials=overwrite_existing_partials,
                                       chunk_size=chunk_size)


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

    # Either log to file (silent console) or to console with Rich
    if log_file is None:
        handlers = [RichHandler(show_time=True, show_level=False)]
    else:
        handlers = [logging.FileHandler(log_file, mode='a')]

    # Configure logger
    log_format = '%(asctime)s %(levelname)s %(threadName)s [%(module)s.%(funcName)s] %(message)s'
    logging.basicConfig(level=log_level, datefmt='%Y-%m-%d %H:%M:%S', format=log_format, handlers=handlers)


def check_chunks(chunks, pipeline_config):
    if chunks < 1:
        raise RuntimeError('You must have at least 1 chunk')

    append_enabled = pipeline_config['steps']['output'].get('append', True)
    if chunks > 1 and append_enabled:
        raise RuntimeError('Output mode must be set to "append = false" in "output" step when using more than one chunk')


if __name__ == "__main__":
    app()
