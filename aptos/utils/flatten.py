from pathlib import Path


class CodeExtractor:
    """
    Produces a flat version of this package, so it can be copy-pasted into a notebook.
    """

    ignore = [
        'setup.py',
        '__init__.py',
        'cli.py',
        'flatten.py',
        'adaptive.py',
        'adaptive_test.py',
        'cubic_spline.py',
        'cubic_spline_test.py',
        'distribution.py',
        'distribution_test.py',
        'general.py',
        'general_test.py',
        'util.py',
        'util_test.py',
        'wavelet.py',
        'wavelet_test.py',
        'visualization.py',
    ]

    def __init__(self, dest='data/flat-code.txt'):
        self.root = Path.cwd() / 'aptos'
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


class ImportExtractor:
    """
    Produces a flat version of this package, so it can be copy-pasted into a notebook.
    """

    ignore = [
        'setup.py',
        '__init__.py',
        'cli.py',
        'flatten.py',
        'adaptive.py',
        'adaptive_test.py',
        'cubic_spline.py',
        'cubic_spline_test.py',
        'distribution.py',
        'distribution_test.py',
        'general.py',
        'general_test.py',
        'util.py',
        'util_test.py',
        'wavelet.py',
        'wavelet_test.py'
    ]

    def __init__(self, dest='data/flat-imports.txt'):
        self.root = Path.cwd() / 'aptos'
        self.dest = Path(dest)

    def start(self):
        lines = []
        for imports in self.yield_imports():
            lines.extend(imports)
        lines = set(lines)
        lines = list(lines)
        lines.sort(reverse=True)
        with open(self.dest, 'w') as fh:
            fh.writelines(lines)

    def yield_imports(self):
        for py in self.root.glob('**/*.py'):
            check_ignore = [i in py.name for i in self.ignore]
            if any(check_ignore):
                continue
            with open(py) as fh:
                lines = fh.readlines()

            import_lines = [l for l in lines
                          if l.startswith('from') or l.startswith('import')]
            external_import_lines = [l for l in import_lines
                                     if not l.startswith('from .') and
                                        not l.startswith('import .') and
                                        not l.startswith('from aptos') and
                                        not l.startswith('import aptos') and
                                        not l.startswith('from __future__')]
            yield external_import_lines


if __name__ == '__main__':
    CodeExtractor().start()
    ImportExtractor().start()
