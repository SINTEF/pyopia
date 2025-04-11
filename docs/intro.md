Welcome!
==================================

This is documentation for PyOPIA: Python Ocean Particle Image Analysis, hosted by [SINTEF Ocean](https://www.sintef.no/en/ocean/), in collaboration with University of Plymouth, NTNU, The MBA, George Washington University, and National Oceanography Center (Southampton). We hope to encourage wider collaboration on analysis tools for ocean particles and welcome new contributors!

PyOPIA started in Feb. 2022 as a 'spin-off' from elements of [PySilCam](https://github.com/SINTEF/PySilCam/wiki) that could be relevant for other hardware and image types, including holography.

The code repository for PyOPIA can be found [here](https://github.com/SINTEF/pyopia/).

Pipelines
==================================
PyOPIA aims to provide a pipeline-based workflow as a standard for analysis of particle images in the ocean.

This pipeline should be consistent across different instruments (hardware), and therefore has flexibility to adapt analysis steps to meet instrument-specific processing needs (i.e. holographic reconstruction), while maintaining a traceable workflow that is attached as metadata to a standard output file (HDF5) that helps users follow FAIR data principles.

See {class}`pyopia.pipeline.Pipeline` for more details and examples of how to process images with PyOPIA.

A function-based toolbox
==================================

PyOPIA tools are organised into the following modules:

* background correction {mod}`pyopia.background`.
* processing {mod}`pyopia.process`.
* statistical analysis {mod}`pyopia.statistics`.
* classification {mod}`pyopia.classify`.
* metadata & datafile formatting {mod}`pyopia.io`.

You can combine these tools for exploring different analysis approaches (i.e. in notebooks).
We hope this can help more exploratory development and contributions to the PyOPIA code.

If you are analysing data for publication, we recommend using the {class}`pyopia.pipeline.Pipeline` standard so that your analysis steps are documented and the output format is more easily shareable.

Full documentation for the code is [here](api)

Installing
==================================

Users are expected to be familiar with Python and [uv](https://docs.astral.sh/uv/getting-started/installation/). You can create a new uv project and install PyOPIA like this:

```
uv init --python 3.12 mypyopiaproject
cd mypyopiaproject
uv add pyopia[classification]
```

To run PyOPIA, either use uv (uv run pyopia --help), or activate the venv first (source .venv/bin/activate), before running pyopia (pyopia --help).

The [classification] part installs tensorflow which is required by PyOPIA's Classification module, and is optional. 

To confirm that everything was installed correctly, you can run a PyOPIA test for the classifier:

```
uv run python -m pyopia.tests.test_classify
```

If you would like to install a development environment, please refer to the instructions in the README on GitHub, [here](https://github.com/SINTEF/pyopia/blob/main/README.md)

Links to libraries PyOPIA uses
==================================

PyOPIA is a high-level tool that makes use of several open source libraries. Please see the list of libraries listed in the pyproject.toml file if you are interested in what is used.

Processing and plotting modeuls makes routine use of several functions provided by libraries including: [scikit-image](https://scikit-image.org/),
[numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [xarray](https://docs.xarray.dev), [matplotlib](https://matplotlib.org/),
[cmocean](https://matplotlib.org/cmocean/), [tensorflow](https://www.tensorflow.org/), and [keras](https://keras.io/).
