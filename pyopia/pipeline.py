class Pipeline():
    
    def __init__(self, steps):
        self.steps = steps
        pass
    
    def run(self, raw_image):
        
        imbw = self.steps['segmentation'](raw_image)
        
        stats = self.steps['statextract'](imbw)
            
        metastring = str(self.steps)
        
        # write metastring to h5 file