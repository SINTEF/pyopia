'''
Module for managing the PyOpia processing pipeline

Refer to :class:`Pipeline` for examples of how to process datasets and images
'''


class Pipeline():
    '''The processing pipeline class
    ================================

    Examples:
    ^^^^^^^^^

    A holographic processing pipeline:
    """"""""""""""""""""""""""""""""""

    .. code-block:: python

        datafile_hdf = 'proc/holotest'
        model_path = exampledata.get_example_model()
        threshold = 0.9

        holo_common_settings = {'pixel_size': 4.4, # pixel size in um
                                'wavelength': 658, # laser wavelength in nm
                                'minZ': 22, # minimum reconstruction distance in mm
                                'maxZ': 60, # maximum reconstruction distance in mm
                                'stepZ': 2} #step size in mm

        steps = {'common': holo.Common('imbg.pgm', **holo_common_settings),
                'classifier': Classify(model_path=model_path),
                'load': holo.Load(filename),
                'imageprep': holo.Reconstruct(stack_clean=0),
                'statextract': CalculateStats(threshold=threshold),
                'output': pyopia.io.StatsH5(datafile_hdf)}

        processing_pipeline = Pipeline(steps)

    A silcam processing pipeline:
    """""""""""""""""""""""""""""

    .. code-block:: python

        datafile_hdf = 'proc/test'
        model_path = exampledata.get_example_model()
        threshold = 0.85

        steps = {'common': Common(),
                'load': SilCamLoad(filename),
                'classifier': Classify(model_path=model_path),
                'imageprep': ImagePrep(),
                'statextract': CalculateStats(threshold=threshold),
                'output': pyopia.io.StatsH5(datafile_hdf)}

        processing_pipeline = Pipeline(steps)

    Running a pipeline:
    """""""""""""""""""

    .. code-block:: python

        stats = processing_pipeline.run()


    You can check the workflow used by reading the steps from the metadata in output file, like this:

    .. code-block:: python

        pyopia.io.show_h5_meta(datafile_hdf + '-STATS.h5')

    '''

    def __init__(self, steps):
        self.steps = steps
        self.common = self.steps['common']()
        self.cl = self.steps['classifier']()
        self.cl.load_model()

    def run(self):
        '''Method for executing the processing pipeline

        Returns:
            stats (DataFrame): stats DataFrame of particle statistics
        '''

        timestamp, imraw = self.steps['load']()
        imc = self.steps['imageprep'](imraw, self.common)
        stats = self.steps['statextract'](timestamp, imc, self.cl)

        self.steps['output'](stats, steps_to_string(self.steps))

        return stats

    def print_steps(self):
        '''Print the steps dictionary
        '''

        # an eventual metadata parser could replace this below printing
        # and format into an appropriate standard
        print('\n-- Pipeline configuration --\n')
        from pyopia import __version__ as pyopia_version
        print('PyOpia version: ' + pyopia_version + '\n')
        print(steps_to_string(self.steps))
        print('\n---------------------------------\n')


def steps_to_string(steps):
    '''Convert pipeline steps dictionary to a human-readable string

    Args:
        steps (dict): pipeline steps dictionary

    Returns:
        str: human-readable string of the types and variables
    '''

    steps_str = '\n'
    for i, key in enumerate(steps.keys()):
        steps_str += (str(i + 1) + ') Step: ' + key
                      + '\n   Type: ' + str(type(steps[key]))
                      + '\n   Vars: ' + str(vars(steps[key]))
                      + '\n')
    return steps_str


class Common():
    '''Class for routines that are common for all images,
    and only run onece at the start processing (e.g. creation of a static baground image)

    This is the default Common class that just returns an empty dict

    If returning a background image, this could be contained in the output dictionary as 'imbg'

    An example cutomised Common class is :class:`pyopia.instrument.holo.Common`
    '''
    def __init__(self):
        pass

    def __call__(self):
        return dict()
