"""Exercise the real image-picker callback and inference in a hidden Tk window."""

import argparse
from pathlib import Path
import sys
import time
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from app import F1ClassifierApp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = tk.Tk()
    root.withdraw()
    try:
        app = F1ClassifierApp(root)
        app.load_model(args.checkpoint)
        # Supply a path at the dialog boundary; all preview and inference code is real.
        with patch('app.filedialog.askopenfilename', return_value=str(args.image.resolve())):
            app.choose_image()
        deadline = time.monotonic() + 30
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert not app.busy, 'GUI inference did not complete within 30 seconds.'
        assert app.photo is not None, 'Image preview was not created.'
        assert len(app.table.get_children()) == 3, 'Expected three ranked predictions.'
        assert app.status.get().startswith('Prediction:'), app.status.get()
        print(app.status.get())
        for row in app.table.get_children():
            print(app.table.item(row)['values'])
        print('GUI image selection, preview, worker inference and table update passed.')
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
