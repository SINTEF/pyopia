class Pipeline():

    def __init__(self, steps):
        self.steps = steps
        self.cl = self.steps['classifier']()
        self.cl.load_model()
        self.datafile_hdf = 'proc/test'
        pass

    def run(self):

        timestamp, imc = self.steps['load']()

        stats, imbw, saturation = self.steps['statextract'](timestamp, imc, self.cl)
        
        stats['timestamp'] = timestamp
        stats['saturation'] = saturation

        # stats = self.steps['process'](cl, data)
        
        self.steps['output'](self.datafile_hdf, stats, steps_to_string(self.steps))
        print('output done.')

        return stats

        # write metastring to h5 file

    def print_steps(self):

        from pyopia import __version__ as pyopia_version

        # an eventual metadata parser could replace this below loop
        # and format into an appropriate standard
        print('\n-- Pipeline configuration --\n')
        print('PyOpia version: ' + pyopia_version + '\n')
        print(steps_to_string(self.steps))
        print('\n---------------------------------\n')


def steps_to_string(steps):
    steps_str = ''
    for key in steps.keys():
        steps_str += ('Step name: ' + key + ':\n'
                        + str(type(steps[key]))
                        + '\n' + str(vars(steps[key])))
    return steps_str