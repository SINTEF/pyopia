'''
PyOPIA top-level code primarily for managing cmd line entry points
'''

import typer

app = typer.Typer()


@app.command()
def docs():
    '''Open browser at PyOPIA's readthedocs page
    '''
    print("Opening PyOPIA's docs")
    typer.launch("https://pyopia.readthedocs.io")


if __name__ == "__main__":
    app()
