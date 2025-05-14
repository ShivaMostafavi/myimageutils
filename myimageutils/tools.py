import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import generic_filter
import scipy.io as sio
from pathlib import Path
import PerfusionImaging.tool as perfusion
import ants as ants
import mplcursors 
import SimpleITK as sitk
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from ipywidgets import interact
from ipywidgets.widgets import IntSlider
from scipy import ndimage
import pydicom
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from skimage.metrics import structural_similarity 
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from collections import Counter


def load_mat_legacy(path):
    """
    Returns a dict {variable_name: numpy_array}
    for a classic (<v7.3) MATLAB .mat file.
    """
    path = Path(path).expanduser()
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    # squeeze_me=True  → drop length‑1 dimensions
    # struct_as_record=False  → MATLAB structs become simple objects
    return {k: v for k, v in data.items() if not k.startswith("__")}



def mask_fun(img):
    x = img[:].copy()
    x[x >= 1] = 1
    x[x < 0] = 0
    return x

def list_display(img_list, name="", vmax=300, cmap='jet', read_dcm=False, titles=None):
    """
    Display a list of images with interactive slice control and optional subplot titles.

    Parameters:
        img_list: list of 3D numpy arrays or DICOM file paths
        name: overall figure title
        vmax: color intensity cap
        cmap: colormap for display
        read_dcm: if True, img_list contains DICOM file paths
        titles: list of titles for each subplot
    """
    if read_dcm:
        img_list = [sitk.GetArrayFromImage(sitk.ReadImage(dcm_file)) for dcm_file in img_list]

    max_slices_contrast = max(img.shape[2] for img in img_list)

    contrast_slice_slider = widgets.IntSlider(min=0, max=max_slices_contrast - 1, step=1, value=0, description='Slice:')

    def display_slice(contrast_slice_index, img_list, name):
        n = len(img_list)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        fig.suptitle(name + f'  Slice {contrast_slice_index}')
        if n == 1:
            axes = [axes]  # ensure it's iterable

        for i, img in enumerate(img_list):
            if img is not None:
                slice_idx = min(img.shape[2] - 1, contrast_slice_index)
                axes[i].imshow(img[:, :, slice_idx], cmap=cmap, vmin=0, vmax=vmax)
                axes[i].axis('off')
                if titles and i < len(titles):
                    axes[i].set_title(titles[i], fontsize=12)
        plt.show()

    def update(contrast_slice_index):
        display_slice(contrast_slice_index, img_list, name)

    widgets.interact(update, contrast_slice_index=contrast_slice_slider)
    
def Avg_flow(dcm_rest, dcm_mask_rest, rest_auc, hu_value = 45,tissue_rho=1.053):
    dcm_rest[~dcm_mask_rest[:].astype(bool)] = np.nan
    A = np.mean(dcm_rest[dcm_mask_rest])-(hu_value)
    voxel_size = dcm_rest.spacing
    organ_mass = (
        tissue_rho *
        voxel_size[0] *
        voxel_size[1] *
        voxel_size[2] / 1000
        )
    Q_avg = A/(rest_auc*organ_mass)
    return Q_avg



def plot_gamma_curve(time, value, title="γ‑variate fit"):
    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        result = perfusion.gamma_plot(ax, time, value)[0]
    except Exception as e:
        ax.text(0.05, 0.5, f"gamma_plot failed:\n{e}",
                ha="left", va="center", fontsize=12)
        ax.axis("off")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration / a.u.")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return result



def get_mask_img(img, mask):
    # Ensure img is a NumPy array
    fix_array = np.array(img[:], copy=True)

    # Apply the mask (only modify valid data)
    fix_array[mask[:] == 0] = 0  

    # Convert to ANTsImage
    fix_img = ants.from_numpy(fix_array)

    # Preserve original metadata (spacing, origin, direction)
    fix_img.set_spacing(img.spacing)
    fix_img.set_origin(img.origin)
    fix_img.set_direction(img.direction)

    return fix_img

def erode_3d(mask, size=2):
    """
    Perform binary erosion on a 3D binary mask.
    
    Parameters
    ----------
    mask : np.ndarray
        A 3D binary array (shape e.g. (512, 512, 320)).
    size : int
        Erosion size (radius).
    
    Returns
    -------
    np.ndarray
        The eroded binary mask, same shape as input.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {mask.shape}")

    structure = np.ones((2*size+1,) * 3)  # 3D cube kernel
    
    # Check shape (just for logging or sanity)
    print(f"Input shape: {mask.shape}")
    print(f"Structure shape: {structure.shape}")
    
    # Apply erosion directly (assuming axis order is fine)
    eroded = ndimage.binary_erosion(mask, structure=structure).astype(mask.dtype)

    # Sanity check
    if eroded.shape != mask.shape:
        raise RuntimeError(f"Shape mismatch: eroded {eroded.shape}, original {mask.shape}")
    
    return eroded

def nanmedian_filter3d(img, size=3):
    def nanmedian(values):
        return np.nanmedian(values)
    filtered = generic_filter(img, nanmedian, size=(size, size, size), mode='nearest')
    return filtered
def nanmean_filter3d(img, size=3):
    def nanmean(values):
        return np.nanmean(values)
    filtered = generic_filter(img, nanmean, size=(size, size, size), mode='nearest')
    return filtered
def nanmax_filter3d(img, size=3):
    def nanmax(values):
        return np.nanmax(values)
    filtered = generic_filter(img, nanmax, size=(size, size, size), mode='nearest')
    return filtered


def classify_3d_cfr_stress(cfr_map, stress_map, show_plot=True):
    """
    Classifies 3D CFR/stress voxel maps into predefined regions and plots a histogram with percentages.

    Parameters:
        cfr_map (3D np.ndarray): CFR values.
        stress_map (3D np.ndarray): Stress perfusion values.
        show_plot (bool): Whether to display a histogram.

    Returns:
        dict: Region name → count of voxels.
        3D np.ndarray of region labels.
    """
    if cfr_map.shape != stress_map.shape:
        raise ValueError("CFR and stress maps must have the same shape.")

    # Define classification regions
    regions = {
        "Normal Flow": Polygon([(3.37/5, 3.37), (2.39, 3.37), (2.39, 2.39/2), (10, 5), (10, 10), (2, 10)]),
        "Minimally Reduced": Polygon([(1.76, 0.88), (1.76, 2.7), (0.54, 2.7), (3.37/5, 3.37), (2.39, 3.37), (2.39, 2.39/2)]),
        "Mildly Reduced": Polygon([(0.406, 2.03), (1.12, 2.03), (1.12, 0.56), (1.76, 0.88), (1.76, 2.7), (0.54, 2.7)]),
        "Moderately Reduced Flow Capacity": Polygon([(1.74/5, 1.74), (0.91, 1.74), (0.91, 0.91/2), (1.12, 0.56), (1.12, 2.03), (0.406, 2.03)]),
        "Myocardial Steal": Polygon([(1/5, 1), (1.74/5, 1.74), (0.91, 1.74), (0.91, 1)]),
        "Definite Ischemia": Polygon([(0, 0), (1/5, 1), (0.91, 1), (0.91, 0.91/2)]),
        "Predominantly Transmural Myocardial Scar": Polygon([(0, 0), (2, 10), (0, 10)]),
    }

    region_colors = {
        "Normal Flow": "#FF0000",
        "Minimally Reduced": "#FF8C00",
        "Mildly Reduced": "#FFD700",
        "Moderately Reduced Flow Capacity": "#008000",
        "Myocardial Steal": "#4B0082",
        "Definite Ischemia": "#0000FF",
        "Predominantly Transmural Myocardial Scar": "#000000",
        "Unclassified": "gray"
    }

    # Initialize output arrays
    labels_3d = np.full(cfr_map.shape, fill_value='Unclassified', dtype=object)
    flat_labels = []

    # Loop through all voxels
    # Loop through all voxels
    for idx in np.ndindex(cfr_map.shape):
        cfr = cfr_map[idx]
        stress = stress_map[idx]

        # Skip invalid data
        if np.isnan(cfr) or np.isnan(stress):
            continue  # Don't classify or add to flat_labels

        point = Point(stress, cfr)
        classified = False
        for label, poly in regions.items():
            if poly.contains(point):
                labels_3d[idx] = label
                flat_labels.append(label)
                classified = True
                break

        # if not classified:
        #     labels_3d[idx] = "Unclassified"
        #     flat_labels.append("Unclassified")


    # Count occurrences
    counts = Counter(flat_labels)
    total = sum(counts.values())
        # Fixed label order
    label_order = [
        "Normal Flow",
        "Minimally Reduced",
        "Mildly Reduced",
        "Moderately Reduced Flow Capacity",
        "Myocardial Steal",
        "Definite Ischemia",
        "Predominantly Transmural Myocardial Scar"
    ]

    # Make sure all labels are represented, even with 0 count
    ordered_counts = {label: counts.get(label, 0) for label in label_order}
    total = sum(ordered_counts.values())

    # Plot
    if show_plot:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            ordered_counts.keys(),
            ordered_counts.values(),
            color=[region_colors[label] for label in label_order]
        )
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Region")
        plt.ylabel("Number of Voxels")
        plt.title("Voxel Distribution by CFR Region")

        # Add percentage labels
        for bar, label in zip(bars, label_order):
            height = bar.get_height()
            percent = (height / total * 100) if total > 0 else 0
            plt.text(bar.get_x() + bar.get_width()/2, height + 5,
                     f"{percent:.1f}%", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()
    # Plot
    if show_plot:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(counts.keys(), counts.values(),
                       color=[region_colors.get(label, 'gray') for label in counts.keys()])
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Region")
        plt.ylabel("Number of Voxels")
        plt.title("Voxel Distribution by CFR Region")

        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            percent = (height / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2, height + 5, f"{percent:.1f}%", 
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    return dict(counts), labels_3d

