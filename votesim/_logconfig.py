# -*- coding: utf-8 -*-

"""
Configure module logger. 

"""
import logging
import os
logger = logging.getLogger(__name__)



class LogSettings(object):
    """Module debug logger"""
    def __init__(self):
        return
    
    
    def start_debug(self, 
                    filename='votesim.log',
                    fullpath='', 
                    level=logging.DEBUG):
        
        if fullpath == '':
            dirname = os.path.dirname(__file__)
            path  = os.path.join(dirname, filename)
        else:
            path = fullpath
            
        logging.basicConfig(filename=path, level=level)
        logger.info('log path = %s', path)
        
        self.path = path
        self.logger = logger
        return path
    
    
    def start_warn(self,
                   filename='votesim.log',
                   fullpath='', 
                   level=logging.WARNING):
        
        return self.start_debug(filename, fullpath, level)
    
    
    def print(self):
        with open(self.path, 'r') as f:
            for line in f:
                print(line, end='')
        return
    
    def delete(self):
        logging.shutdown()
        os.remove(self.path)








# def startlogger(filename='votesim.log', level='INFO', display=False):
#     """
#     Start module logger.
    
#     Parameters
#     -----------
#     filename : str or None (default)
#         Name of file to write.
#     level : str
#         Logging level
#     display : bool (default True)
#         Display to screen if true
#     """
#     #logger = logging.getLogger(__name__)
#     try:
#         level = level.upper()
#         level = getattr(logging, level)
#     except AttributeError:
#         pass
    
#     handlers = []
    
#     if filename is not None:    
#         fh = logging.FileHandler(filename)
#         fh.name = 'TextLogger'
#         handlers.append(fh)
    
#     if display:
#         ch = logging.StreamHandler()
#         ch.name = 'StreamLogger'
#         handlers.append(ch)
        
#     logging.basicConfig(handlers=handlers, level=level)
        
#     return 


# def log2file(filename = 'votesim.log', **kwargs):
    
#     if filename is not None:    
#         fh = logging.FileHandler(filename)
#         fh.name = 'TextLogger'
#         logger.addHandler(fh)
        
        
        
# def setDebug():
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
#     return

# def setInfo():
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)

# def setWarning():
#     logger = logging.getLogger()
#     logger.setLevel(logging.WARNING)
                    
