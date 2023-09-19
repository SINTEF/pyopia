'''
PyOPIA top-level code primarily for managing cmd line entry points
'''

import typer
import toml

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
            from pyopia.instrument.silcam import generate_config
        case 'holo':
            from pyopia.instrument.holo import generate_config

    pipeline_config = generate_config(raw_files, model_path, outfolder, output_prefix)
    config_filename = instrument + "-config.toml"
    with open(config_filename, "w") as toml_file:
        toml.dump(pipeline_config, toml_file)


if __name__ == "__main__":
    app()
