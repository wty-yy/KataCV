import contextlib, time

class Stopwatch(contextlib.ContextDecorator):
    def __init__(self, t=0.0):
        self.t = t
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, type, value, traceback):
        self.dt = time.time() - self.start
        self.t += self.dt
