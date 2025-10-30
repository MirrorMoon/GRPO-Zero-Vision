"""Vision-specific dataset utilities and reward function for GRPO."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_types import TensorMiniBatch


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.BICUBIC)
        array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


class VisionClassificationDataset(Dataset):
    """Simple image-folder dataset for classification tasks."""

    def __init__(
        self,
        data_dir: str,
        image_size: int,
        augment: bool = False,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        self.image_size = image_size
        self.augment = augment
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.samples: List[Tuple[Path, int]] = []
        self.class_names: List[str] = []

        self._scan()

    def _scan(self) -> None:
        class_dirs = [p for p in self.data_dir.iterdir() if p.is_dir()]
        class_dirs.sort()
        if not class_dirs:
            raise ValueError(f"No class subdirectories found in {self.data_dir}")
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for label, class_dir in enumerate(class_dirs):
            self.class_names.append(class_dir.name)
            for ext in exts:
                for path in sorted(class_dir.rglob(ext)):
                    self.samples.append((path, label))
        if not self.samples:
            raise ValueError(f"No images found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return image
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        return image

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = _load_image(path, self.image_size)
        image = self._apply_augmentation(image)
        image = (image - self.mean) / self.std
        metadata = {"path": str(path), "label": int(label)}
        return image, label, metadata

    @staticmethod
    def collate_fn(batch: Iterable[Tuple[torch.Tensor, int, Dict]]) -> TensorMiniBatch:
        images, labels, metadata = zip(*batch)
        batch_images = torch.stack(list(images), dim=0)
        batch_labels = torch.tensor(labels, dtype=torch.long)
        group_ids = [item["path"] for item in metadata]
        return TensorMiniBatch(
            inputs=batch_images,
            targets=batch_labels,
            group_ids=group_ids,
            metadata=list(metadata),
        )


def classification_reward(
    response: torch.Tensor, target: torch.Tensor, sample_metadata: Dict
) -> Dict[str, Any]:
    if response.dim() > 1:
        response = response.squeeze(0)
    if isinstance(target, torch.Tensor):
        target_index = int(target.item())
    else:
        target_index = int(target)
    predicted_index = int(torch.argmax(response).item())
    reward = 1.0 if predicted_index == target_index else 0.0
    reward_info = {
        "accuracy": reward,
        "predicted_index": predicted_index,
        "target_index": target_index,
    }
    policy_metadata = {"action_index": predicted_index}
    return {"reward": reward, "reward_info": reward_info, "policy_metadata": policy_metadata}

