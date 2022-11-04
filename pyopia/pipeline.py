
class Pipeline():

    def __init__(self, steps):
        self.steps = steps
        self.cl = self.steps['classifier']()
        self.cl.load_model()
        pass

    def run(self):

        timestamp, imc = self.steps['load']()

        stats, imbw, saturation = self.steps['statextract'](timestamp, imc, self.cl)

        #stats = self.steps['process'](cl, data)

        return stats

        # write metastring to h5 file

    def print_steps(self):

        from pyopia import __version__ as pyopia_version

        # an eventual metadata parser could replace this below loop
        # and format into an appropriate standard
        print('\n-- Pipeline configuration --\n')
        print('PyOpia version: ' + pyopia_version + '\n')
        for key in self.steps.keys():
            print('Step name: ' + key + ':\n'
                  + str(type(self.steps[key]))
                  + '\n' + str(vars(self.steps[key]))
                  + '\n')
        print('---------------------------------\n')
