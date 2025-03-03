import os

from torch.utils.data import Dataset
from PIL import Image


class ImageFolderDataset(Dataset):
    """
    Пользовательский датасет для папки с изображениями.
    Все файлы с расширениями .png, .jpg, .jpeg, .bmp будут загружены.
    Так как данные не размечены, возвращается фиктивная метка (0).
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): путь к папке с изображениями.
            transform (callable, optional): Трансформации, которые применяются к изображению.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            os.path.join(root_dir, file)
            for file in os.listdir(root_dir)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        self.image_files = sorted(self.image_files)  # Опционально сортируем файлы

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Возвращаем изображение и фиктивную метку
        return image, 0