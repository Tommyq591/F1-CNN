"""Run this file in PyCharm to choose a photo and predict its F1 team."""

import json
import pickle
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk
import torch

from f1_cnn.predict import TeamPredictor


PROJECT = Path(__file__).resolve().parent


class F1ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.predictor = None
        self.results = queue.Queue()
        self.photo = None
        self.busy = False
        root.title('F1 Team Classifier')
        root.geometry('900x760')
        root.minsize(650, 600)
        frame = ttk.Frame(root, padding=22)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='F1 Team Classifier', font=('Segoe UI', 24, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='Choose a photo with one main F1 car to see the predicted team.').pack(anchor='w', pady=(4, 14))
        self.model_text = tk.StringVar(value='No trained model loaded')
        ttk.Label(frame, textvariable=self.model_text, wraplength=810).pack(anchor='w')
        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=12)
        self.choose_button = ttk.Button(controls, text='Choose image', command=self.choose_image)
        self.choose_button.pack(side='left')
        self.model_button = ttk.Button(controls, text='Load model', command=self.choose_model)
        self.model_button.pack(side='left', padx=10)
        self.preview = ttk.Label(frame, anchor='center')
        self.preview.pack(fill='both', expand=True)
        self.status = tk.StringVar(value='Load a checkpoint or train with: python -m f1_cnn train')
        ttk.Label(frame, textvariable=self.status, font=('Segoe UI', 14, 'bold'), wraplength=810).pack(anchor='w', pady=12)
        self.table = ttk.Treeview(frame, columns=('team', 'score'), show='headings', height=3)
        self.table.heading('team', text='Team')
        self.table.heading('score', text='Model score')
        self.table.column('team', width=550)
        self.table.column('score', width=170)
        self.table.pack(fill='x')
        ttk.Label(frame, text='Scores are not calibrated certainty. The model always chooses a known team,\n'
                  'even for unrelated photos. It does not locate cars or separate multiple teams.',
                  wraplength=810).pack(anchor='w', pady=(14, 0))
        pointer = PROJECT / 'runs' / 'latest.json'
        if pointer.exists():
            try:
                self.load_model(json.loads(pointer.read_text(encoding='utf-8'))['checkpoint'])
            except (OSError, ValueError, RuntimeError, KeyError, pickle.UnpicklingError) as error:
                self.status.set(f'Could not load the last model: {error}')
        root.after(100, self.poll)

    def load_model(self, path):
        predictor = TeamPredictor(path)
        self.predictor = predictor
        self.model_text.set(f'Model: {Path(path).name} | {len(predictor.class_names)} teams | {Path(path).parent.name}')
        self.status.set('Ready. Choose an image.')

    def choose_model(self):
        path = filedialog.askopenfilename(title='Choose a trained checkpoint', initialdir=PROJECT / 'runs',
                                          filetypes=[('PyTorch checkpoint', '*.pt')])
        if path:
            try:
                self.load_model(path)
            except (OSError, ValueError, RuntimeError, KeyError, pickle.UnpicklingError) as error:
                messagebox.showerror('Cannot load model', str(error))

    def choose_image(self):
        if self.predictor is None:
            messagebox.showinfo('Train or load a model', 'Run python -m f1_cnn train, then load its best.pt checkpoint.')
            return
        path = filedialog.askopenfilename(title='Choose an F1 photo',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.webp *.bmp'), ('All files', '*.*')])
        if not path:
            return
        try:
            with Image.open(path) as source:
                preview = ImageOps.exif_transpose(source).convert('RGB')
            preview.thumbnail((810, 390))
            self.photo = ImageTk.PhotoImage(preview)
            self.preview.configure(image=self.photo)
        except (OSError, ValueError) as error:
            messagebox.showerror('Cannot read image', str(error))
            return
        for item in self.table.get_children():
            self.table.delete(item)
        self.status.set('Classifying...')
        self.busy = True
        self.choose_button.state(['disabled'])
        self.model_button.state(['disabled'])
        # Inference runs off the GUI thread; widgets are updated only by poll().
        threading.Thread(target=self.classify, args=(path,), daemon=True).start()

    def classify(self, path):
        try:
            self.results.put(('ok', self.predictor.predict(path)))
        except Exception as error:
            self.results.put(('error', str(error)))

    def poll(self):
        try:
            kind, result = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.choose_button.state(['!disabled'])
            self.model_button.state(['!disabled'])
            if kind == 'ok':
                self.status.set(f"Prediction: {result[0]['team']}")
                for row in result:
                    self.table.insert('', 'end', values=(row['team'], f"{row['score']:.1%}"))
            else:
                self.status.set('Prediction failed')
                messagebox.showerror('Prediction failed', result)
        self.root.after(100, self.poll)


def main():
    torch.set_num_threads(2)
    root = tk.Tk()
    F1ClassifierApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
