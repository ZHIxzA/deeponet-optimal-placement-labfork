from skimage.restoration import denoise_tv_chambolle
import torch
from torchvision import transforms

class AnisotropicDiffusion:
    def __init__(self, weight=0.1):
        """
        Custom PyTorch-compatible transform for anisotropic diffusion.
        :param weight: Regularization weight for anisotropic diffusion.
        """
        self.weight = weight

    def __call__(self, img):
        """
        Applies anisotropic diffusion on the input image.
        :param img: Input image as a PIL Image or NumPy array.
        :return: Processed image as a NumPy array.
        """
        # Convert the image to NumPy array if it is a PIL Image
        if isinstance(img, torch.Tensor):
            img = img.numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        processed_img = denoise_tv_chambolle(img, weight=self.weight,)
        return torch.tensor(processed_img.transpose(2, 0, 1))  # Convert back to (C, H, W)


from scipy.ndimage import median_filter

class MedianFilter:
    def __init__(self, size=3):
        self.size = size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        processed_img = median_filter(img, size=self.size)
        return torch.tensor(processed_img)