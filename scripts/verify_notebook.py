"""Execute the saved-results notebook in a fresh kernel and retain its real outputs."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# All runtime files remain writable in restricted Windows environments.
for variable, folder in [('IPYTHONDIR', 'ipython'), ('JUPYTER_CONFIG_DIR', 'jupyter_config'),
                         ('JUPYTER_RUNTIME_DIR', 'jupyter_runtime'), ('MPLCONFIGDIR', 'matplotlib')]:
    location = ROOT / '.cache' / folder
    location.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(location)

import nbformat
from nbclient import NotebookClient


def main():
    path = ROOT / 'F1_CNN.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=600, startup_timeout=180, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    # Only record success after all code cells have actually executed without an error.
    lines = notebook.cells[0].source.splitlines()
    lines[1] = '# Execution: all cells ran without error in a fresh local kernel; RUN_TRAINING=False displays saved results.'
    notebook.cells[0].source = '\n'.join(lines) + '\n'
    for name in ['F1_CNN.ipynb', 'F1_CNN.pynb.ipynb']:
        nbformat.write(notebook, ROOT / name)
    print('Both notebooks saved with verified execution and outputs.', flush=True)


if __name__ == '__main__':
    main()
