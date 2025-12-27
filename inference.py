import torch
import torchvision.transforms as T
import numpy as np
import cv2
from utils import refine_mask
from u2net import U2NET

class SegmentationModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = U2NET(3, 1)
        self.model.load_state_dict(
            torch.load("u2net.pth", map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def segment_person(self, image):
        w, h = image.size

        inp = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(inp)[0]

        # Ensure (H, W)
        pred = pred.squeeze().cpu().numpy()

        # Normalize to [0,255]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        mask = (pred * 255).astype(np.uint8)

        # Resize to original image size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Refine edges (hair-friendly)
        mask = refine_mask(mask)

        return mask
