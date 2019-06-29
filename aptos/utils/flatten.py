from pathlib import Path


class CodeExtractor:
    """
    Produces a flat version of this package, so it can be copy-pasted into a notebook.
    """

    ignore = [
        'setup.py',
        '__init__.py',
        'cli.py'
    ]

    def __init__(self, dest='data/flat-code.txt'):
        self.root = Path.cwd()
        self.dest = Path(dest)

    def start(self):
        with open(self.dest, 'w') as fh:
            for f, c in self.yield_code():
                fh.write(f'# -- {f} -- START --\n')
                fh.writelines(c)
                fh.write(f'\n# -- {f} -- END --\n\n')

    def yield_code(self):
        for py in self.root.glob('**/*.py'):
            check_ignore = [i in py.name for i in self.ignore]
            if any(check_ignore):
                continue
            with open(py) as fh:
                lines = fh.readlines()

            code_lines = [l for l in lines
                          if not l.startswith('from') and not l.startswith('import')]
            yield py, code_lines

if __name__ == '__main__':
    ce = CodeExtractor()
    ce.start()