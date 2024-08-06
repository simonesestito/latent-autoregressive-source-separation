import torch
import torchmetrics
import torchvision
import tqdm
import pathlib
from typing import Generator

l2 = torch.nn.MSELoss()
psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)

def measure_l2(gt_image: torch.Tensor, gen_image: torch.Tensor):
    gt_images, gen_images = gt_image.unsqueeze(0), gen_image.unsqueeze(0)
    assert gt_images.shape == gen_images.shape
    assert gt_images.shape[0] == gen_images.shape[0] == 1
    return l2(gt_image.unsqueeze(0), gen_image.unsqueeze(0))


def measure_psnr(gt_image: torch.Tensor, gen_image: torch.Tensor):
    gt_images, gen_images = gt_image.unsqueeze(0), gen_image.unsqueeze(0)
    assert gt_images.shape == gen_images.shape
    assert gt_images.shape[0] == gen_images.shape[0] == 1
    return psnr(gt_image.unsqueeze(0), gen_image.unsqueeze(0))


def get_images(dir: pathlib.Path) -> Generator[tuple[tuple[pathlib.Path, pathlib.Path], tuple[pathlib.Path, pathlib.Path]], None, None]:
    original_subdir, separated_subdir = dir / "ori", dir / "sep"

    # Get all images in the ori/ subdirectory
    for original_1_image_path in original_subdir.glob("*-1.png"):
        # Extract the image index
        image_index = int(original_1_image_path.stem.split("-")[0])

        # Get the corresponding images tuple
        original_2_image_path = original_subdir / f"{image_index}-2.png"
        separated_1_image_path = separated_subdir / f"{image_index}-1.png"
        separated_2_image_path = separated_subdir / f"{image_index}-2.png"

        yield (original_1_image_path, original_2_image_path), (separated_1_image_path, separated_2_image_path)


def load_images(dir: pathlib.Path) -> Generator[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]], None, None]:
    for (original_1_image_path, original_2_image_path), (separated_1_image_path, separated_2_image_path) in get_images(dir):
        yield (
            (
                torchvision.io.read_image(str(original_1_image_path)).float() / 255.0,
                torchvision.io.read_image(str(original_2_image_path)).float() / 255.0
            ),
            (
                torchvision.io.read_image(str(separated_1_image_path)).float() / 255.0,
                torchvision.io.read_image(str(separated_2_image_path)).float() / 255.0
            )
        )


def compute_metrics(root_dir: pathlib.Path):
    avg_l2, avg_psnr, images_count = 0, 0, 0
    for (original_1_image, original_2_image), (separated_1_image, separated_2_image) in tqdm.tqdm(load_images(root_dir), total=10_000):
        image_1_l2 = measure_l2(original_1_image, separated_1_image)
        image_2_l2 = measure_l2(original_2_image, separated_2_image)
        image_1_psnr = measure_psnr(original_1_image, separated_1_image)
        image_2_psnr = measure_psnr(original_2_image, separated_2_image)

        avg_l2 += (image_1_l2 + image_2_l2) / 2
        avg_psnr += (image_1_psnr + image_2_psnr) / 2

        images_count += 1

    avg_l2 /= images_count
    avg_psnr /= images_count

    return avg_l2, avg_psnr


def main():
    root_dirs = [
        pathlib.Path("separated-images-TEACHER/"),
        pathlib.Path("separated-images-STUDENT-FULL/"),
        pathlib.Path("separated-images-STUDENT-ALONE/"),
    ]

    for root_dir in root_dirs:
        avg_l2, avg_psnr = compute_metrics(root_dir)
        print(f"Root directory: {root_dir}")
        print(f"Average L2: {avg_l2:.5f}")
        print(f"Average PSNR: {avg_psnr:.5f}")
        print("-" * 80)


if __name__ == "__main__":
    main()