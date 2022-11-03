
class custom_statextract():
    
    def __eval__(self):
        '''do something illigal'''
        return


steps = {'load': load_image(threshold=0.85),
         'statextract': custom_statextract(min_area=12),
         'data_output','h5'}

proc = Pipeline(steps)

proc.run()

------

class Pipeline():
    
    def __init__(self, steps) -> None:
        self.steps = steps
        pass
    
    def run(self, raw_image):
        
        imbw = self.steps['segmentation'](raw_image)
        
        stats = self.steps['statextract'](imbw)
            
        metastring = str(steps)
        
        # write metastring to h5 file
        
        


def segmentation_maker(threshold):
    def segmentation_fast(img):
        return img>threshold
    return segmentation_fast
steps = {'segmentation': segmentation_maker(threshold=0.5)}


class SegmentationFast():
    
    def __init__(self, threshold):
        self.threshold = threshold
        pass
    
    def __eval__(self, img):
        imbw = img>self.threshold
        return imbw
        
        
class SegmentationSlow():
    
    def run():
        
        


class ExportParticle():


class StatExtract():
    
    def __init__(self,
                 default_var1=1) -> None:
        self.default_var1 = default_var1
        pass
    
    def run():
        steps()
        steps1()
        steps2()
    
    def steps():
        
        
    def step1()
    
    
    def step2()