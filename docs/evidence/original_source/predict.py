from PIL import Image, ImageOps
import torch

from .data import F1ImageDataset
from .model import F1TeamCNN


class TeamPredictor:
    """Classify one photo using saved class order and evaluation preprocessing."""

    def __init__(self, checkpoint, device='cpu'):
        self.device = torch.device(device)
        # weights_only avoids executing arbitrary pickled classes when restoring tensors.
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.class_names = saved['class_names']
        self.model = F1TeamCNN(len(self.class_names), saved['config']['dropout']).to(self.device)
        self.model.load_state_dict(saved['state_dict'])
        self.model.eval()  # Disable dropout for repeatable predictions.
        self.transform = F1ImageDataset([], saved['config']['image_size']).transform

    def predict(self, image_path, top_k=3):
        if top_k < 1:
            raise ValueError('top_k must be positive.')
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        with torch.no_grad():
            logits = self.model(self.transform(image).unsqueeze(0).to(self.device))
            scores, indices = logits.softmax(1)[0].topk(min(top_k, len(self.class_names)))
        # Softmax scores are model scores, not calibrated certainty.
        return [{'team': self.class_names[i], 'score': float(score)}
                for score, i in zip(scores.cpu(), indices.cpu().tolist())]
