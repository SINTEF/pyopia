
class Pipeline():
    
    def __init__(self, steps):
        self.steps = steps
        pass
    
    def run(self):
        
        data = self.steps['load']()
        
        cl = self.steps['classifier']()
        
        cl.load_model()
        
        stats = self.steps['process'](cl, data)
        
        return stats
        
        # write metastring to h5 file