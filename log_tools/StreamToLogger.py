import sys

# From https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
class StreamToLogger(object):
    def __init__(self, logger, level, background = True):
        self.logger = logger
        self.level = level
        self.linebuf = ''
        self.background = background

    def write (self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
            if not self.background:
                sys.__stdout__.write(line + '\n')
    
    def flush(self):
        pass