import sys
import re

class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d remaining'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=', output=sys.stderr, init=""):
        assert len(symbol) == 1
        print(init)
        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d', r'\g<name>%dd' % len(str(total)), fmt)
        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'
        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        shown = self.fmt
        if (remaining == 0):
            shown = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) Done!!!          '       
        print('\r' + shown % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output )