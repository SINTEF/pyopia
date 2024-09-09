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
        raw_files = FilesToProcess(pipeline_config['general']['raw_files'])
        if 'correctbackground' in pipeline_config['steps']:
            raw_files.get_static_background_files(average_window=pipeline_config['steps']['correctbackground']['average_window'])
        raw_files.chunk_files(chunks)

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

        def process_file_list(file_list, c):
            processing_pipeline = Pipeline(pipeline_config)
            for filename in track(file_list, description=f'[blue]Processing progress (thread {c})',
                                  disable=c != 0):
                try:
                    processing_pipeline.run(filename)
                except Exception as e:
                    progress.console.print("[red]An error occured in processing, " +
                                           "skipping rest of pipeline and moving to next image.")
                    logger.error(e)
                    logger.debug(''.join(traceback.format_tb(e.__traceback__)))

        for c, chunk in enumerate(raw_files.chunked_files):
            job = threading.Thread(target=process_file_list, args=(chunk, c, ))
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


class FilesToProcess:
    def __init__(self, glob_pattern=None):
        '''Build file list from glob pattern if specified.
           Create FilesToProcess.chunked_files is chunks specified
           File list from glob will be sorted.

        Parameters
        ----------
        glob_pattern : str, optional
            Glob pattern, by default None
        '''
        self.files = None
        self.background_files = []
        if glob_pattern is not None:
            self.files = sorted(glob(glob_pattern))

    def from_filelist_file(self, path_to_filelist):
        '''
        Initialize explicit list of files to process from a text file.
        The text file should contain one path to an image per line, which should be processed in order.
        '''
        with open(path_to_filelist, 'r') as fh:
            self.files = list(fh.readlines())

    def to_filelist_file(self, path_to_filelist):
        '''Write file list to a txt file

        Parameters
        ----------
        path_to_filelist : str
            Path to txt file to write
        '''
        with open(path_to_filelist, 'w') as fh:
            [fh.writelines(L + '\n') for L in self.files]

    def chunk_files(self, chunks):
        '''Chunk the file list and create FilesToProcess.chunked_files

        Parameters
        ----------
        chunks : int
            number of chunks to produce (must be at least 1)
        '''
        assert chunks > 0, 'you must have at least one chunk'
        n = int(np.ceil(len(self.files) / chunks))
        self.chunked_files = [self.files[i:i + n] for i in range(0, len(self.files), n)]

    def insert_bg_files_into_chunks(self, bgshift_function='pass'):
        if bgshift_function == 'pass':
            for c in self.chunked_files:
                # we have to loop backwards over bg_files because we are inserting into the top of the chunk
                c = [c.insert(0, bg_file) for bg_file in reversed(self.background_files)]
        else:
            raise Exception('not implemented')

    def get_static_background_files(self, average_window=0):
        self.background_files = []
        for f in self.files[0:average_window]:
            self.background_files.append(f)

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for filename in self.files:
            yield filename


if __name__ == "__main__":
    app()
