from torchvision import transforms as T
from typing import Tuple, Callable

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD  = [0.229, 0.224, 0.225]


class SSLPairTransform:
    def __init__(self, img_size: int = 224,
                 mean: Tuple[float, float, float] = DEFAULT_MEAN,
                 std: Tuple[float, float, float]  = DEFAULT_STD
        ):
        self.img_size = img_size
        self._mean, self._std = mean, std

        self._strong_ops = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0),
                                ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        self._base_ops = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    # --------- интерфейс ----------
    def strong(self, img):
        return self._strong_ops(img)

    def base(self, img):
        return self._base_ops(img)

    def __call__(self, img):
        return self.strong(img), self.strong(img)
