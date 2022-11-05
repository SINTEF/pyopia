class Pipeline():
    '''Processing pipeline class

    further explanation @todo
    '''

    def __init__(self, steps):
        self.steps = steps
        self.cl = self.steps['classifier']()
        self.cl.load_model()

    def run(self):
        '''Method for executing the processing pipeline

        Returns:
            stats (DataFrame): stats DataFrame of particle statistics
        '''

        timestamp, imc = self.steps['load']()

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
