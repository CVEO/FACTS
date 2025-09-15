import cv2
import numpy as np
from .node import BaseNode
from . import utils


# --- Merged functions from cielab_auto_processor.py ---

def harmonize_brightness_iterative(img_list, threshold=2.0, max_iter=50):
    """
    Improved iterative mean target brightness balancing - gentle adjustment version.
    """
    if not img_list:
        print("No images to process.")
        return

    print(
        f"Starting improved iterative mean target brightness balancing (Threshold: {threshold}, Max Iterations: {max_iter})")

    # 1. Convert to LAB color space and calculate initial L* means
    lab_images = {img.name: cv2.cvtColor(utils.pil_to_cv2_bgr(img.img_data), cv2.COLOR_BGR2LAB) for img in img_list}
    l_means = {name: np.mean(lab.astype(np.float64)[:, :, 0]) for name, lab in lab_images.items()}

    # 2. Use the median as the reference brightness for robustness
    global_reference_l = np.median(list(l_means.values()))
    print(f"   Global reference L* value: {global_reference_l:.2f} (using median)")

    # 3. Gentle iterative adjustment
    for i in range(max_iter):
        deviations = {name: abs(mean - global_reference_l) for name, mean in l_means.items()}
        max_dev_name = max(deviations, key=deviations.get)
        max_deviation = deviations[max_dev_name]

        if max_deviation < threshold:
            print(f"   Convergence reached after {i + 1} iterations. Max deviation: {max_deviation:.2f}")
            break

        current_lab = lab_images[max_dev_name].astype(np.float64)
        current_mean = l_means[max_dev_name]

        target_adjustment = global_reference_l - current_mean
        max_adjustment = min(abs(target_adjustment), 20)
        adjustment = np.sign(target_adjustment) * max_adjustment * 0.5

        current_lab[:, :, 0] += adjustment
        np.clip(current_lab[:, :, 0], 0, 255, out=current_lab[:, :, 0])

        lab_images[max_dev_name] = current_lab.astype(np.uint8)
        l_means[max_dev_name] = np.mean(current_lab[:, :, 0])

    # 4. Convert back to BGR and update img_data
    img_map = {img.name: img for img in img_list}
    for name, lab_img in lab_images.items():
        bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        img_map[name].img_data = utils.cv2_bgr_to_pil(bgr_img)
        print(f"  - Harmonized {name}")

    print(f"   Finished harmonization for {len(lab_images)} images.")


def harmonize_brightness_statistical(img_list, use_clahe=False, epsilon=1e-6):
    """
    Improved global statistical brightness normalization - conservative adjustment version.
    """
    if not img_list:
        print("No images to process.")
        return

    print(
        f"Starting improved global statistical brightness normalization (CLAHE: {'Enabled' if use_clahe else 'Disabled'})")

    lab_images = {img.name: cv2.cvtColor(utils.pil_to_cv2_bgr(img.img_data), cv2.COLOR_BGR2LAB) for img in img_list}

    means = []
    stds = []
    names_order = list(lab_images.keys())
    for name in names_order:
        l_channel = lab_images[name].astype(np.float64)[:, :, 0]
        means.append(np.mean(l_channel))
        stds.append(np.std(l_channel))

    mu_global = np.median(means)
    sigma_global = np.median(stds)

    mean_range = np.percentile(means, 75) - np.percentile(means, 25)
    adjustment_factor = min(1.0, 30.0 / mean_range) if mean_range > 0 else 0.3

    print(f"   Global statistics: μ_global={mu_global:.2f}, σ_global={sigma_global:.2f}")
    print(f"   Adjustment factor: {adjustment_factor:.3f} (conservative adjustment)")

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) if use_clahe else None

    img_map = {img.name: img for img in img_list}
    for i, name in enumerate(names_order):
        lab_img = lab_images[name]
        l_channel = lab_img.astype(np.float64)[:, :, 0]
        mu_k = means[i]
        sigma_k = stds[i]

        if sigma_k > epsilon:
            l_target = mu_global + (l_channel - mu_k) * (sigma_global / sigma_k)
            l_prime_k = l_channel + (l_target - l_channel) * adjustment_factor
            adjustment_diff = l_prime_k - l_channel
            max_pixel_adjustment = 40
            adjustment_diff = np.clip(adjustment_diff, -max_pixel_adjustment, max_pixel_adjustment)
            l_prime_k = l_channel + adjustment_diff
        else:
            l_prime_k = l_channel

        np.clip(l_prime_k, 0, 255, out=l_prime_k)
        lab_img[:, :, 0] = l_prime_k.astype(np.uint8)

        if clahe:
            lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])

        bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        img_map[name].img_data = utils.cv2_bgr_to_pil(bgr_img)
        print(f"  - Harmonized {name}")

    print(f"   Finished harmonization for {len(lab_images)} images.")


def harmonize_brightness_adaptive(img_list):
    """
    Adaptive illumination equalization algorithm - intelligent adjustment based on image content.
    """
    if not img_list:
        print("No images to process.")
        return

    print(f"Starting adaptive illumination equalization")

    lab_images = {img.name: cv2.cvtColor(utils.pil_to_cv2_bgr(img.img_data), cv2.COLOR_BGR2LAB) for img in img_list}

    image_stats = {}
    for name, lab_img in lab_images.items():
        l_channel = lab_img.astype(np.float64)[:, :, 0]
        mean_l = np.mean(l_channel)
        hist = np.histogram(l_channel, bins=256, range=(0, 255))[0]
        dark_ratio = np.sum(hist[:85]) / np.sum(hist)
        bright_ratio = np.sum(hist[171:]) / np.sum(hist)

        image_stats[name] = {
            'mean': mean_l,
            'std': np.std(l_channel),
            'dark_ratio': dark_ratio,
            'bright_ratio': bright_ratio,
            'needs_major_adjustment': dark_ratio > 0.7 or bright_ratio > 0.7 or mean_l < 60 or mean_l > 180
        }

    normal_means = [stats['mean'] for stats in image_stats.values() if not stats['needs_major_adjustment']]
    target_mean = np.median(normal_means) if normal_means else np.median([s['mean'] for s in image_stats.values()])

    print(f"   Target brightness: {target_mean:.2f}")

    img_map = {img.name: img for img in img_list}
    for name, lab_img in lab_images.items():
        l_channel = lab_img.astype(np.float64)[:, :, 0]
        stats = image_stats[name]

        if stats['needs_major_adjustment']:
            adjustment = (target_mean - stats['mean']) * 0.3
            l_channel += adjustment
        else:
            kernel = np.ones((15, 15), np.float32) / 225
            local_mean = cv2.filter2D(l_channel.astype(np.float32), -1, kernel)
            global_diff = target_mean - stats['mean']
            local_adjustment = global_diff * 0.2 * (1 - np.abs(l_channel - local_mean) / 128)
            l_channel += local_adjustment

        np.clip(l_channel, 0, 255, out=l_channel)
        lab_img[:, :, 0] = l_channel.astype(np.uint8)

        bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        img_map[name].img_data = utils.cv2_bgr_to_pil(bgr_img)
        print(f"  - Harmonized {name}")

    print(f"   Finished adaptive adjustment for {len(lab_images)} images.")


class Illumination(BaseNode):
    """
    A workflow node for advanced brightness normalization across a set of images.
    It uses various CIELAB-based algorithms to make illumination consistent.
    
    Methods:
    - 'adaptive': (Default) Smart adjustment based on image content. Recommended.
    - 'iterative': Gently adjusts images one by one towards a median brightness.
    - 'statistical': A conservative version of statistical normalization.
    """

    def __init__(self, image_processors, method='adaptive', **kwargs):
        super().__init__(image_processors)
        self.method = method
        self.kwargs = kwargs

    def process(self):
        """
        Executes the brightness normalization process on the images.
        """
        if len(self.image_processors) < 2:
            print("Only one image provided, no illumination balancing needed.")
            return

        print(f"---'Running Illumination Normalization (Method: {self.method})'---")

        if self.method == 'adaptive':
            harmonize_brightness_adaptive(self.image_processors)
        elif self.method == 'iterative':
            harmonize_brightness_iterative(self.image_processors, **self.kwargs)
        elif self.method == 'statistical':
            harmonize_brightness_statistical(self.image_processors, **self.kwargs)
        else:
            print(f"Unknown illumination method: {self.method}. Using 'adaptive'.")
            harmonize_brightness_adaptive(self.image_processors)

        print("---'Illumination Normalization Complete'---")
