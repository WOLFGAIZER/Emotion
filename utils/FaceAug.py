import cv2
import numpy as np
import random
import os

class FaceAugmentor:
    def __init__(self, save_dir="generated_faces"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def add_glasses(self, image):
        """Simulate glasses by drawing semi-transparent rectangles over eyes."""
        h, w = image.shape[:2]
        overlay = image.copy()
        y1, y2 = int(h * 0.35), int(h * 0.50)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return cv2.addWeighted(overlay, 0.3, image, 0.7, 0)


    def change_hair_style(self, image, messy=True):
        """Simulate hair style by modifying top area texture."""
        h, w = image.shape[:2]
        overlay = image.copy()
        y1, y2 = 0, int(h * 0.25)
        noise = np.random.randint(0, 80 if messy else 30, (y2 - y1, w, 3), dtype=np.uint8)
        overlay[y1:y2, :] = cv2.add(overlay[y1:y2, :], noise)
        return cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    def vary_lighting(self, image):
        """Randomly brighten or darken image."""
        factor = random.uniform(0.6, 1.4)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return image

    def tilt_head(self, image):
        """Rotate image slightly to simulate head tilt."""
        h, w = image.shape[:2]
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))

    def generate_variants(self, face_image, target_id="unknown"):
        """Generate multiple variants of a given face."""
        variants = []
        output_dir = os.path.join(self.save_dir, target_id)
        os.makedirs(output_dir, exist_ok=True)

        transformations = [
            ("glasses", self.add_glasses),
            ("messy_hair", lambda img: self.change_hair_style(img, messy=True)),
            ("combed_hair", lambda img: self.change_hair_style(img, messy=False)),
            ("lighting", self.vary_lighting),
            ("tilt", self.tilt_head)
        ]

        for name, func in transformations:
            aug_img = func(face_image)
            path = os.path.join(output_dir, f"{name}.jpg")
            cv2.imwrite(path, aug_img)
            variants.append(aug_img)

        return variants
