import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import generic_filter
from scipy.io import sio
from pathlib import Path



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


