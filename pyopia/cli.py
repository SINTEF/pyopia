"""
PyOPIA top-level code primarily for managing cmd line entry points
"""

import typer
import toml
import os
import time
import datetime
import traceback
import logging
import json
import numpy as np
from rich.progress import Progress
from rich.logging import RichHandler
from rich import print
import rich.progress
import pandas as pd
import multiprocessing
import pathlib
from pathlib import Path
import contextlib
import matplotlib.pyplot as plt
import zipfile
from typing import Tuple
from typing_extensions import Annotated
from operator import methodcaller
import sys

import pyopia
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
import pyopia.auxillarydata
import pyopia.exampledata
import pyopia.metadata

app = typer.Typer()


# Create a callback function for --version
def version_callback(value: bool):
    if value:
        typer.echo(f"PyOPIA version: {pyopia.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the application's version and exit.",
    )
):
    pass  # This is needed to allow version to be a global option


@app.command()
def docs():
    """Open browser at PyOPIA's readthedocs page"""
    print("Opening PyOPIA's docs")
    typer.launch("https://pyopia.readthedocs.io")


@app.command()
def modify_config(
    existing_filename: str,
    modified_filename: str,
    raw_files=None,
    pixel_size=None,
    step_name=None,
    modify_arg=None,
    modify_value=None,
):
    """Modify a existing config.toml file and write a new one to disc

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
    """
    toml_settings = pyopia.io.load_toml(existing_filename)

    if raw_files is not None:
        toml_settings["general"]["raw_files"] = f"{raw_files}"
    if pixel_size is not None:
        toml_settings["general"]["pixel_size"] = float(pixel_size)

    if step_name is not None:
        try:
            if modify_arg == "average_window":
                modify_value = int(modify_value)
            elif modify_arg == "threshold":
                modify_value = float(modify_value)
            else:
                modify_value = str(modify_value)
        except ValueError:
            pass

        toml_settings["steps"][step_name][modify_arg] = modify_value

    with open(modified_filename, "w") as toml_file:
        toml.dump(toml_settings, toml_file)


@app.command()
def generate_config(
    instrument: str, raw_files: str, model_path: str, outfolder: str, output_prefix: str
):
    """Put an example config.toml file in the current directory

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
    """
    match instrument:
        case "silcam":
            pipeline_config = pyopia.instrument.silcam.generate_config(
                raw_files, model_path, outfolder, output_prefix
            )
        case "holo":
            pipeline_config = pyopia.instrument.holo.generate_config(
                raw_files, model_path, outfolder, output_prefix
            )
        case "uvp":
            pipeline_config = pyopia.instrument.uvp.generate_config(
                raw_files, model_path, outfolder, output_prefix
            )

    config_filename = instrument + "-config.toml"
    with open(config_filename, "w") as toml_file:
        toml.dump(pipeline_config, toml_file)


@app.command()
def init_project(
    project_name: str, instrument: str = "silcam", example_data: bool = False
):
    """Initialize a PyOPIA processing project with a standard config file and folder layout

    Parameters
    ----------
    project_name : str
        Name of project, a folder with this name will be created
    instrument : str
        Either `silcam`, `holo` or `uvp`
    example_data: bool
        If specified, download 10 example SilCam images and put them in the images/ folder
    """
    raw_files = "images/*.silc"
    outfolder = "processed"
    output_prefix = project_name
    model_path = ""
    config_filename = "config.toml"
    metadata_file_name = "metadata.json"
    proj_folder = pathlib.Path(project_name)
    auxdata_file_name = "auxillary_data.csv"
    auxdata_folder_name = "auxillarydata"
    auxdata_folder = pathlib.Path(proj_folder, auxdata_folder_name)
    auxillary_data_path = pathlib.Path(auxdata_folder, auxdata_file_name)

    title = "NOT_SPECIFIED"
    longitude = latitude = np.nan
    if example_data:
        title = "PyOPIA example data"
        longitude = 14.45498
        latitude = 68.89363
        instrument = "silcam"

    # @todo Move this to instrument modules
    seavox_instrument_identifier = (
        "SDN:L22::TOOL2206" if instrument == "silcam" else "NOT_SPECIFIED"
    )

    project_metadata_template = pyopia.metadata.Metadata(
        title=title,
        project_name=project_name,
        longitude=longitude,
        latitude=latitude,
        instrument=instrument,
        seavox_instrument_identifier=seavox_instrument_identifier,
    )

    readme_lines = [
        f"{project_name}",
        "=" * len(project_name),
        "@TODO: Place your .silc images in the images/ folder",
        "@TODO: Describe your project and data in this README file",
        "@TODO: Update metadata.txt with project info",
        "@TODO: Update auxillary_data.csv with per-image data",
        "@TODO: Update PyOPIA config.toml as needed - note that logging to pyopia.log file is enabled",
    ]

    # Create project folder
    # If it exists, exist with warning
    print(f"[blue]Creating PyOPIA project folder {proj_folder}")
    if proj_folder.exists():
        print(f"[red]ERROR: Project folder {proj_folder} exists")
        return
    proj_folder.mkdir()

    # Create subfolders
    print("[blue]Creating project folder structure")
    auxdata_folder.mkdir()
    images_folder = pathlib.Path(proj_folder, "images")
    images_folder.mkdir()

    # Get default classifier
    print("[blue]Downloading PyOPIA example classifier")
    with contextlib.chdir(proj_folder):
        model_path = pyopia.exampledata.get_example_model()

    # Generate config
    print("[blue]Generating default PyOPIA config")
    config_generator = getattr(pyopia.instrument, instrument).generate_config
    pipeline_config = config_generator(raw_files, model_path, outfolder, output_prefix)
    pipeline_config["steps"]["output"]["project_metadata_file"] = metadata_file_name
    pipeline_config["steps"]["output"]["auxillary_data_file"] = (
        f"{auxdata_folder_name}/{auxdata_file_name}"
    )
    with open(pathlib.Path(proj_folder, config_filename), "w") as toml_file:
        toml.dump(pipeline_config, toml_file)

    # Create directory for processed files
    (
        proj_folder / Path(pipeline_config["steps"]["output"]["output_datafile"]).parent
    ).mkdir()

    # Add README file
    print("[blue]Adding README file")
    with open(pathlib.Path(proj_folder, "README"), "w") as fh:
        # print(*readme_lines, sep="\n", end="\n", file=fh)
        fh.writelines(line + "\n" for line in readme_lines)

    # Generate project metadata template file
    print("[blue]Creating metadata template file")
    with open(pathlib.Path(proj_folder, metadata_file_name), "w") as fh:
        json.dump(project_metadata_template.model_dump(), fh, indent=4)

    # Generate auxillary data template file
    with open(auxillary_data_path, "w") as fh:
        fh.write(pyopia.auxillarydata.AUXILLARY_DATA_FILE_TEMPLATE)

    # Get example image data
    if example_data:
        print("[blue]Downloading example SilCam images")
        example_imgs_zip = proj_folder / Path("pyopia-example-images-10.zip")
        pyopia.exampledata.get_file_from_pysilcam_blob(
            example_imgs_zip.name, download_directory=example_imgs_zip.parent
        )
        with zipfile.ZipFile(example_imgs_zip, "r") as zipit:
            zipit.extractall(proj_folder / Path("images/"))
        os.remove(example_imgs_zip)


@app.command()
def process(config_filename: str, num_chunks: int = 1, strategy: str = "block"):
    """Run a PyOPIA processing pipeline based on given a config.toml

    Parameters
    ----------
    config_filename : str
        Config filename

    numchunks : int, optional
        Split the dataset into chucks, and process in parallell, by default 1

    strategy : str, optional
        Strategy to use for chunking dataset, either `block` or `interleave`. Defult: `block`
    """
    t1 = time.time()

    with Progress(transient=True) as progress:
        progress.console.print(f"[blue]PYOPIA VERSION {pyopia.__version__}")

        progress.console.print("[blue]LOAD CONFIG")
        pipeline_config = pyopia.io.load_toml(config_filename)

        setup_logging(pipeline_config)
        logger = logging.getLogger("rich")
        logger.info(f"PyOPIA process started {pd.Timestamp.now()}")

        check_chunks(num_chunks, pipeline_config)

        progress.console.print("[blue]OBTAIN IMAGE LIST")
        conf_corrbg = pipeline_config["steps"].get("correctbackground", dict())
        average_window = conf_corrbg.get("average_window", 0)
        bgshift_function = conf_corrbg.get("bgshift_function", "pass")
        raw_files = pyopia.pipeline.FilesToProcess(
            pipeline_config["general"]["raw_files"]
        )
        raw_files.prepare_chunking(
            num_chunks, average_window, bgshift_function, strategy=strategy
        )

        # Write the dataset list of images to a text file
        raw_files.to_filelist_file("filelist.txt")

        progress.console.print("[blue]PREPARE FOLDERS")
        if "output" not in pipeline_config["steps"]:
            raise Exception(
                'The given config file is missing an "output" step.\n'
                + "This is needed to setup how to save data to disc."
            )
        output_datafile = pipeline_config["steps"]["output"]["output_datafile"]
        os.makedirs(os.path.split(output_datafile)[:-1][0], exist_ok=True)

        if os.path.isfile(output_datafile + "-STATS.nc"):
            dt_now = datetime.datetime.now().strftime("D%Y%m%dT%H%M%S")
            newname = output_datafile + "-conflict-" + str(dt_now) + "-STATS.nc"
            logger.warning(f"Renaming conflicting file to: {newname}")
            os.rename(output_datafile + "-STATS.nc", newname)

        progress.console.print("[blue]INITIALISE PIPELINE")

    # With one chunk we keep the non-multiprocess functionality to ensure backwards compatibility
    job_list = []
    if num_chunks == 1:
        process_file_list(raw_files, pipeline_config, 0)
    else:
        for c, chunk in enumerate(raw_files.chunked_files):
            job = multiprocessing.Process(
                target=process_file_list, args=(chunk, pipeline_config, c)
            )
            job_list.append(job)

    # Start all the jobs
    [job.start() for job in job_list]

    # If we are using multiprocessing, make sure all jobs have finished
    [job.join() for job in job_list]

    # Calculate and print total processing time
    time_total = pd.to_timedelta(time.time() - t1, "seconds")
    with Progress(transient=True) as progress:
        progress.console.print(f"[blue]PROCESSING COMPLETED IN {time_total}")


@app.command()
def merge_mfdata(
    path_to_data: str,
    prefix="*",
    overwrite_existing_partials: bool = True,
    chunk_size: int = None,
):
    """Combine a multi-file directory of STATS.nc files into a single '-STATS.nc' file
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
    """
    setup_logging({"general": {}})

    pyopia.io.merge_and_save_mfdataset(
        path_to_data,
        prefix=prefix,
        overwrite_existing_partials=overwrite_existing_partials,
        chunk_size=chunk_size,
    )


@app.command()
def convert_raw_images(config_filename: str):
    """Convert raw images files to bitmap format (png).

    Input images are inferred from config.toml file. Ouput folder is created.

    Parameters
    ----------
    config_filename : str
        Config filename
    """
    output_folder = pathlib.Path("images_converted")
    print(f"[blue]CREATING IMAGE OUTPUT FOLDER: {output_folder}")
    output_folder.mkdir()

    print("[blue]LOAD CONFIG")
    config = pyopia.io.load_toml(config_filename)

    print("[blue]GENERATING IMAGE FILE LIST")
    p = pathlib.Path(config["general"]["raw_files"])
    image_files = list(pathlib.Path(p.parent).glob(p.name))

    # Initialize the image loading function based on config specification
    loader_class = config["steps"]["load"]["pipeline_class"]
    classname = loader_class.split(".")[-1]
    modulename = ".".join(loader_class.split(".")[:-1])
    keys = [k for k in config["steps"]["load"] if k != "pipeline_class"]
    arguments = {k: config["steps"]["load"] for k in keys}
    m = methodcaller(classname, **arguments)
    callobj = m(sys.modules[modulename])

    # Loop over image list, load, convert, store
    for filename in rich.progress.track(
        image_files, description="Converting raw images"
    ):
        data = dict(filename=filename)
        callobj(data)
        plt.imsave(pathlib.Path(output_folder, filename.stem + ".png"), data["imraw"])


@app.command()
def make_montage(
    stats_filename: pathlib.Path,
    output_filename: str = "montage.png",
    filter_variable: Annotated[Tuple[str, float, float], typer.Option()] = [
        None,
        None,
        None,
    ],
):
    """Create a montage of particles

    Parameters
    ----------
    config_filename : str
        Config filename
    output_filename: str
        Store montage figure to this filename
    filter_variable: list
        Variables to filter on (name, min, max), e.g. ['depth', 5, None]
    """
    print("[blue]LOAD STATS")
    xstats = pyopia.io.load_stats(str(stats_filename))
    config = pyopia.io.steps_from_xstats(xstats)

    # Filter the stats
    if filter_variable[0] is not None:
        print(filter_variable)
        xstats = xstats.where(
            (xstats[filter_variable[0]] <= filter_variable[2])
            & (xstats[filter_variable[0]] >= filter_variable[1])
        )

    print("[blue]CREATING MONTAGE")
    montage = pyopia.statistics.make_montage(
        xstats.to_pandas(),
        config["general"]["pixel_size"],
        config["steps"]["statextract"]["export_outputpath"],
        eyecandy=False,
    )

    print("[blue]STORING MONTAGE")
    pyopia.plotting.montage_plot(montage, config["general"]["pixel_size"])
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")


def process_file_list(file_list, pipeline_config, c):
    """Run a PyOPIA processing pipeline for a chuncked list of files based on a given config.toml

    Parameters
    ----------
    file_list : str
        List of file paths to process, where each file will be passed individually through the processing pipeline

    pipeline_config : str
        Loaded config.toml file to initialize the processing pipeline and setup logging

    c : int
        Chunk index for tracking progress and logging. If set to 0, enables the
        progress bar; for other values, the progress bar is disabled.
    """
    processing_pipeline = pyopia.pipeline.Pipeline(pipeline_config)
    setup_logging(pipeline_config)
    logger = logging.getLogger("rich")

    with get_custom_progress_bar(
        f"[blue]Processing progress (chunk {c})", disable=c != 0
    ) as pbar:
        for filename in pbar.track(
            file_list, description=f"[blue]Processing progress (chunk {c})"
        ):
            try:
                logger.debug(f"Chunk {c} starting to process {filename}")
                processing_pipeline.run(filename)
            except Exception as e:
                logger.warning(
                    "[red]An error occured in processing, "
                    + "skipping rest of pipeline and moving to next image."
                    + f"(chunk {c})"
                )
                logger.error(e)
                logger.debug("".join(traceback.format_tb(e.__traceback__)))


def setup_logging(pipeline_config):
    """Configure logging

    Parameters
    ----------
    pipeline_config : dict
        TOML settings
    """
    # Get user parameters or default values for logging
    log_file = pipeline_config["general"].get("log_file", None)
    log_level_name = pipeline_config["general"].get("log_level", "INFO")
    log_level = getattr(logging, log_level_name)

    # Either log to file (silent console) or to console with Rich
    if log_file is None:
        handlers = [RichHandler(show_time=True, show_level=False)]
    else:
        handlers = [logging.FileHandler(log_file, mode="a")]

    # Configure logger
    log_format = "%(asctime)s %(levelname)s %(processName)s [%(module)s.%(funcName)s] %(message)s"
    logging.basicConfig(
        level=log_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format=log_format,
        handlers=handlers,
    )


def check_chunks(chunks, pipeline_config):
    if chunks < 1:
        raise RuntimeError("You must have at least 1 chunk")

    append_enabled = pipeline_config["steps"]["output"].get("append", True)
    if chunks > 1 and append_enabled:
        raise RuntimeError(
            'Output mode must be set to "append = false" in "output" step when using more than one chunk'
        )


def get_custom_progress_bar(description, disable):
    """Create a custom rich.progress.Progress object for displaying progress bars"""
    progress = Progress(
        rich.progress.TextColumn(description),
        rich.progress.BarColumn(),
        rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TextColumn("•"),
        rich.progress.TimeElapsedColumn(),
        rich.progress.TextColumn("•"),
        rich.progress.TimeRemainingColumn(),
        disable=disable,
    )
    return progress


if __name__ == "__main__":
    app()
