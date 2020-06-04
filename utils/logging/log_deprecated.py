'''Module that wraps the logging library from python to make it easier to use.
How tu use:

from log import *

First case: Simple
#Just calling the function without using any configuration

log(message)


Second case: Object
#create a Log() object and change the configuration

logger = Log('/path/to/file.log', level=logging.Warning)
logger.log(message)


Third case: Decorator
# logs all returns when some_func gets called

@log
def some_func():
    do_stuff()


Whenever a Log() object is in scope, the log() (decorator)-function will automatically use it
'''

import logging
import os


class Log():
    '''
    Basic class to use for logging

    Keyword arguments:
    path_to_file -- Path to the file where the logs shall be written in. None for STDOUT ().
    level        -- Level that indicates the priority of the message
                    (Debug 10, Info 20, Warning 30, Error 40, Critical 50)
    '''

    def __init__(self,
                 path_to_file=None,
                 level: int = logging.INFO,
                 message_format: str = '%(levelname)s - %(asctime)s - %(message)s',
                 verbose: bool = False):
        self.path_to_file = os.path.normpath(path_to_file) if path_to_file else None
        self.level = level
        self.message_format = message_format

        logging.basicConfig(filename=self.path_to_file,
                            format=self.message_format,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=self.level)

        self.verbose = verbose

        self.logger = logging.getLogger()
        self.logger.setLevel(self.level)

    def log(self, message: str, level: int = None):
        '''
        Logs a message based on the config of this Log() object.
        '''
        if not level:
            level = self.level
        self.logger.log(level=level, msg=message)

    def set_level(self, level: int):
        '''Sets the level of information that shall be recorded.
        If a message has a lower level it will be ignored.

        Keyword arguments:
        level : int -- (Debug 10, Info 20, Warning 30, Error 40, Critical 50)
        '''
        self.logger.setLevel(level)

    def read_from_file(self, path_to_file: str):
        '''
        TODO
        '''
        pass

    def __repr__(self):
        path = self.path_to_file if self.path_to_file else 'STDOUT'
        representation = f'File: {path}\n'
        representation += f'Level: {logging.getLevelName(self.level)}\n'
        representation += f'Format: {self.message_format}'
        return representation


def _find_logger(verbose: bool = False) -> Log:
    '''
    Checks if there is a Log() Object in local scope an returns it if true.
    Alternatively it creates a new Log() Object and returns that.
    '''
    for var in dir():
        if isinstance(eval(var), Log):
            return eval(var)

    if verbose:
        print('Could not find a Log() object. Creating new one...')
    logger = Log()
    return logger


def log(some_function_or_data, logger: Log = None, verbose: bool = False):
    '''Logs data. Can be used as decorator or just as normal function

    Keyword arguments:
    some_function_or_data -- leave blank for use as decorator. Elsewise the log message
    logger (optional)     -- a Log() element. If not given,
                             it will search in scope for one or create a new one
    '''

    if not isinstance(log, Log):
        logger = _find_logger(verbose)

    # for use as decorator
    if callable(some_function_or_data):
        def wrapper(*args):
            return_value = some_function_or_data(*args)
            logger.log(return_value)
            return return_value

        return wrapper

    # for direct call with message
    logger.log(some_function_or_data)