import os
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.fft import dctn, idctn


class BColors:
    HEADER = '\033[95m'
    OkBLUE = '\033[94m'
    OkCYAN = '\033[96m'
    OkGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ImageUtils:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise FileNotFoundError
        return np.array(Image.open(image_path))

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        max_value = np.max(image)
        min_value = np.min(image)
        if max_value == min_value:
            return np.zeros_like(image)
        return (image - min_value) / (max_value - min_value) * 255

    @staticmethod
    def save_image(image_path: str, image: np.ndarray) -> None:
        if image.dtype != np.uint8:
            # image = ImageUtils.normalize_image(image).astype(np.uint8)
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).save(image_path)


class DCTUtils:
    @staticmethod
    def dctn(array: np.ndarray, axes: tuple, dct_type: int = 2, norm: str = 'ortho') -> np.ndarray:
        return np.array(dctn(array, axes=axes, type=dct_type, norm=norm), dtype=np.float64)

    @staticmethod
    def idctn(array: np.ndarray, axes: tuple, dct_type: int = 2, norm: str = 'ortho') -> np.ndarray:
        return np.array(idctn(array, axes=axes, type=dct_type, norm=norm), dtype=np.float64)

    @staticmethod
    def view_as_blocks(arr_in: np.ndarray, block_shape: tuple) -> np.ndarray:
        h, w = arr_in.shape
        h_pad = h if h % block_shape[0] == 0 else h + (block_shape[0] - h % block_shape[0])
        w_pad = w if w % block_shape[1] == 0 else w + (block_shape[1] - w % block_shape[1])

        if h_pad != h or w_pad != w:
            padded = np.zeros((h_pad, w_pad), dtype=arr_in.dtype)
            padded[:h, :w] = arr_in
            arr_in = padded

        blocks_shape = (arr_in.shape[0] // block_shape[0], arr_in.shape[1] // block_shape[1])
        result = np.zeros(blocks_shape + block_shape, dtype=arr_in.dtype)

        for i in range(blocks_shape[0]):
            for j in range(blocks_shape[1]):
                result[i, j] = arr_in[
                               i * block_shape[0]:(i + 1) * block_shape[0],
                               j * block_shape[1]:(j + 1) * block_shape[1]
                               ]
        return result

    @staticmethod
    def block_dct(image: np.ndarray, block_shape: tuple = (8, 8)) -> np.ndarray:
        blocks = DCTUtils.view_as_blocks(image, block_shape=block_shape)
        return DCTUtils.dctn(blocks, axes=(2, 3))

    @staticmethod
    def block_idct(dct_blocks: np.ndarray) -> np.ndarray:
        recon_blocks = DCTUtils.idctn(dct_blocks, axes=(2, 3))
        rows = [np.concatenate(row_blocks, axis=1) for row_blocks in recon_blocks]
        return np.concatenate(rows, axis=0)


class CompressionUtils:
    @staticmethod
    def create_quantization_matrix(quantization_scale: int = 1) -> np.ndarray:
        return quantization_scale * np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ]
        )

    @staticmethod
    def quantize_image(dct_blocks: np.ndarray, quantization_scale: int = 1) -> np.ndarray:
        quantization_matrix = CompressionUtils.create_quantization_matrix(quantization_scale)
        result = np.zeros(dct_blocks.shape, dtype=dct_blocks.dtype)
        for i in range(dct_blocks.shape[0]):
            for j in range(dct_blocks.shape[1]):
                result[i, j] = np.round(dct_blocks[i, j] / quantization_matrix)
        return result

    @staticmethod
    def dequantize_image(quantized_blocks: np.ndarray, quantization_scale: int = 1) -> np.ndarray:
        quantization_matrix = CompressionUtils.create_quantization_matrix(quantization_scale)
        result = np.zeros(quantized_blocks.shape, dtype=quantized_blocks.dtype)
        for i in range(quantized_blocks.shape[0]):
            for j in range(quantized_blocks.shape[1]):
                result[i, j] = quantized_blocks[i, j] * quantization_matrix
        return result


class Helper:
    @staticmethod
    def create_directories(directories: Dict) -> None:
        for directory in directories.values():
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def calculate_mse(a: np.ndarray, b: np.ndarray) -> float:
        return np.mean((a - b) ** 2)


class Visualization:
    @staticmethod
    def plot_compression_results(
            original_image: np.ndarray,
            results: Dict[str, List],
            quantization_scales: List[int],
            save_path: str = None,
            figsize: Tuple[int, int] = (16, 10)
    ) -> None:

        quantized_images = results["quantized_image"]
        mse_values = results["mse"]

        num_compressed_images = len(quantization_scales)
        images_per_row = 3
        compressed_rows = (num_compressed_images + images_per_row - 1) // images_per_row
        total_rows = compressed_rows + 1

        # Create a figure and axes
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(total_rows, images_per_row)

        # First row, first column: Original image (spans entire row)
        ax_original = fig.add_subplot(gs[0, :])
        ax_original.imshow(original_image, cmap='gray', vmin=0, vmax=255)
        ax_original.set_title('Original Image', fontsize=14)
        ax_original.axis('off')
        ax_original.text(
            10,
            original_image.shape[0] - 10,
            f"MSE: 0.00",
            color='white',
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.7)
        )

        axes = []
        for row in range(1, total_rows):
            for col in range(images_per_row):
                if (row - 1) * images_per_row + col < num_compressed_images:
                    ax = fig.add_subplot(gs[row, col])
                    axes.append(ax)

        for i, (img, mse, q_scale) in enumerate(zip(quantized_images, mse_values, quantization_scales)):
            img_display = img

            axes[i].imshow(img_display, cmap='gray', vmin=0, vmax=255)
            axes[i].set_title(f'Q Scale: {q_scale}', fontsize=12)
            axes[i].axis('off')

            # Add MSE value directly on the image
            axes[i].text(
                10,
                img_display.shape[0] - 10,
                f"MSE: {mse:.2f}",
                color='white',
                fontsize=10,
                bbox=dict(facecolor='black', alpha=0.7)
            )
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.25, wspace=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    directories = {
        "image_base_path": "Images",
        "results": "Images/Results",
    }
    Helper.create_directories(directories)

    original_image = ImageUtils.load_image(
        os.path.join(directories["image_base_path"], "lenna.tif")
    )
    print(BColors.OkGREEN + "Original Image Loaded.\n" + BColors.ENDC, "Original Image Array:\n", original_image)

    original_dct_values = DCTUtils.block_dct(
        image=original_image,
        block_shape=(8, 8)
    )
    print(BColors.OkGREEN + "DCT of Original Image (8x8 Blocks) Calculated.\n" + BColors.ENDC)

    results = {
        "quantized_image": [],
        "mse": []
    }
    quantization_scales = [1, 2, 4, 8, 16, 32]

    for idx, quantization_scale in enumerate(quantization_scales):
        # Step 1: Quantize the DCT coefficients
        quantized_image_dct_values = CompressionUtils.quantize_image(
            dct_blocks=original_dct_values,
            quantization_scale=quantization_scale
        )

        # Step 2: Dequantize (multiply back by quantization matrix)
        dequantized_dct_values = CompressionUtils.dequantize_image(
            quantized_blocks=quantized_image_dct_values,
            quantization_scale=quantization_scale
        )

        # Step 3: Apply inverse DCT to get reconstructed image
        quantized_image = DCTUtils.block_idct(
            dct_blocks=dequantized_dct_values
        )

        results["quantized_image"].append(quantized_image)

        mse = Helper.calculate_mse(original_image, quantized_image)
        results["mse"].append(mse)

        ImageUtils.save_image(
            image_path=os.path.join(directories["results"], f"quantized_lenna_{quantization_scale}.tif"),
            image=quantized_image
        )
        print(
            BColors.WARNING +
            f"Quantization for scale = {quantization_scale}\t\t(MSE = {mse:.2f})\t\t\t\t\t"
            f"{idx+1}/{len(quantization_scales)}" + BColors.ENDC
        )

    print(BColors.OkGREEN + "Quantization Completed for all the Scale factors" + BColors.ENDC)
    Visualization.plot_compression_results(
        original_image=original_image,
        results=results,
        quantization_scales=quantization_scales,
        save_path=os.path.join(directories["results"], "compression_visualization.png"),
        figsize=(8, 8)
    )


if __name__ == '__main__':
    main()