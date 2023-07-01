Welcome!
==================================

This is documentation for PyOpia: Python Ocean Particle Image Analysis, hosted by [SINTEF Ocean](https://www.sintef.no/en/ocean/), in collaboration with University of Plymouth, NTNU, The MBA, and National Oceanography Center (Southampton). We hope to encourage wider collaboration on analysis tools for ocean particles and welcome new contributors!

PyOpia started in Feb. 2022 as a 'spin-off' from elements of [PySilCam](https://github.com/SINTEF/PySilCam/wiki) that could be relevant for other hardware and image types, including holography.

The code repository for PyOpia can be found [here](https://github.com/SINTEF/PyOpia/).

Pipelines
==================================
PyOpia aims to provide a pipeline-based workflow as a standard for analysis of particle images in the ocean.

This pipeline should be consistent across different instruments (hardware), and therefore has flexibility to adapt analysis steps to meet instrument-specific processing needs (i.e. holographic reconstruction), while maintaining a traceable workflow that is attached as metadata to a standard output file (HDF5) that helps users follow FAIR data principles.

See {class}`pyopia.pipeline.Pipeline` for more details and examples of how to process images with PyOpia.

A function-based toolbox
==================================

PyOpia tools are organised into the following modules:

* background correction {mod}`pyopia.background`.
* processing {mod}`pyopia.process`.
* statistical analysis {mod}`pyopia.statistics`.
* classification {mod}`pyopia.classify`.
* metadata & datafile formatting {mod}`pyopia.io`.

You can combine these tools for exploring different analysis approaches (i.e. in notebooks).
We hope this can help more exploratory development and contributions to the PyOpia code.

If you are analysing data for publication, we recommend using the {class}`pyopia.pipeline.Pipeline` standard so that your analysis steps are documented and the output format is more easily shareable.

Full documentation for the code is [here](api)
