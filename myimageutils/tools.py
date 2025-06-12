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


    return dict(counts), labels_3d



def calculate_mean_HU_baseline(dicom_files_ss, dicom_files_v2, seg):

    path_img0 = dicom_files_ss[2]
    ds0 = pydicom.dcmread(path_img0)
    z_info =  ds0.ImagePositionPatient[2]

    tolerance = 1e-3

    # Search through dicom_files_v2 for a matching Z value
    matching_idx = None
    for i, path in enumerate(dicom_files_v2):
        ds1 = pydicom.dcmread(path)
        z1 = ds1.ImagePositionPatient[2]
        if abs(z1 - z_info) < tolerance:
            matching_idx = i
            # print(f"Found match at index {i}, Z = {z1}")
            break

    if matching_idx is None:
        print("No matching slice found within the specified tolerance.")


    ave_HU=[]

    path_img1 = dicom_files_v2[matching_idx]
    ds1 = pydicom.dcmread(path_img1)

    for i in range(len(dicom_files_ss)):
        path_img2 = dicom_files_ss[i]
        ds2 = pydicom.dcmread(path_img2)
        #reading teh mask:
        idx = ds1.InstanceNumber
        temp3 = seg[idx+1]

        temp3 = temp3.pixel_array.astype(bool).transpose(1,0)
        temp3 = np.rot90(temp3, k=1)
        temp3 = temp3[::-1, :]
        structure = np.ones((5, 5), dtype=bool)

        # Apply binary erosion
        eroded_mask = binary_erosion(temp3, structure=structure).astype(np.uint8)
        mask = eroded_mask.copy()



        # === Load images with SimpleITK ===
        img1 = sitk.ReadImage(path_img1)
        img2 = sitk.ReadImage(path_img2)

        # === Extract info for bounds calculation ===
        pos1 = np.array(ds1.ImagePositionPatient)
        pos2 = np.array(ds2.ImagePositionPatient)

        spacing1 = np.array(img1.GetSpacing())
        spacing2 = np.array(img2.GetSpacing())

        size1 = np.array(img1.GetSize())
        size2 = np.array(img2.GetSize())

        # === Calculate physical bounds of each image ===
        bounds1_min = pos1
        bounds1_max = pos1 + spacing1 * (size1 - 1)

        bounds2_min = pos2
        bounds2_max = pos2 + spacing2 * (size2 - 1)

        # print("Bounds img1 min, max:", bounds1_min, bounds1_max)
        # print("Bounds img2 min, max:", bounds2_min, bounds2_max)

        # === Compute overlap bounds in physical space ===
        overlap_min = np.maximum(bounds1_min, bounds2_min)
        overlap_max = np.minimum(bounds1_max, bounds2_max)

        # Validate overlap, allowing Z to be equal for single slices
        overlap_valid = True
        for i in range(len(overlap_min)):
            if i == 2:  # Z axis
                if overlap_max[i] < overlap_min[i]:
                    overlap_valid = False
            else:
                if overlap_max[i] <= overlap_min[i]:
                    overlap_valid = False

        if not overlap_valid:
            raise ValueError(f"No overlap between images! overlap_min={overlap_min}, overlap_max={overlap_max}")

        # print("Overlap min:", overlap_min)
        # print("Overlap max:", overlap_max)

        # === Define new common grid for resampling ===
        new_origin = overlap_min
        new_spacing = np.minimum(spacing1, spacing2)
        new_size = np.ceil((overlap_max - overlap_min) / new_spacing).astype(int)
        new_size = np.maximum(new_size, 1)  # Ensure no zero size dimension

        # For single Z slice, ensure Z size is 1
        if new_size[2] == 0:
            new_size[2] = 1

        # print("New grid origin:", new_origin)
        # print("New grid spacing:", new_spacing)
        # print("New grid size:", new_size)

        # === Check and align image directions ===
        direction1 = np.array(img1.GetDirection())
        direction2 = np.array(img2.GetDirection())
        # print("Direction img1:", direction1)
        # print("Direction img2:", direction2)

        # Use the direction of img1 as the reference for resampling
        new_direction = img1.GetDirection()

        # === Resample image function ===
        def resample_image(image, new_origin, new_spacing, new_size, new_direction):
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputOrigin(new_origin)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size.tolist())
            resampler.SetOutputDirection(new_direction)
            resampler.SetInterpolator(sitk.sitkLinear)  # Switch back to linear for smoother results
            resampler.SetDefaultPixelValue(0)
            resampled = resampler.Execute(image)
            # print(f"Resampled origin: {resampled.GetOrigin()}")
            return resampled

        # === Improved crop_to_overlap function with origin adjustment ===
        def crop_to_overlap(img, overlap_min, overlap_max):
            # Convert physical points to continuous indices (floating-point)
            continuous_index_min = img.TransformPhysicalPointToContinuousIndex(tuple(overlap_min))
            continuous_index_max = img.TransformPhysicalPointToContinuousIndex(tuple(overlap_max))

            # Round indices to nearest integers, ensuring min <= max
            index_min = [int(np.floor(i)) for i in continuous_index_min]
            index_max = [int(np.ceil(i)) for i in continuous_index_max]

            # Clamp indices to valid range
            index_min = [max(0, i) for i in index_min]
            index_max = [min(i, img.GetSize()[d] - 1) for d, i in enumerate(index_max)]

            # Compute size of the region
            size = [max(1, index_max[d] - index_min[d] + 1) for d in range(3)]

            # For single Z slice, ensure size in Z is 1
            if img.GetSize()[2] == 1:
                size[2] = 1
                index_min[2] = 0

            # print(f"Cropping: index_min={index_min}, index_max={index_max}, size={size}")

            # Perform the cropping
            cropped_img = sitk.RegionOfInterest(img, size=size, index=index_min)

            # Compute the new origin of the cropped image (physical position of index_min)
            cropped_origin = np.array(img.TransformIndexToPhysicalPoint(tuple(index_min)))
            # print(f"Cropped origin: {cropped_origin}")

            return cropped_img, cropped_origin

        # Crop images and get their new origins
        img1_cropped, img1_cropped_origin = crop_to_overlap(img1, overlap_min, overlap_max)
        img2_cropped, img2_cropped_origin = crop_to_overlap(img2, overlap_min, overlap_max)

        # Compute the offset between the cropped origins and the desired new_origin
        # new_origin = np.round(new_origin, 5)
        offset1 = img1_cropped_origin - new_origin
        offset2 = img2_cropped_origin - new_origin
        # offset1 = np.round(offset1, 5)
        # offset2 = np.round(offset2, 5)
        # print(f"Offset for img1: {offset1}")
        # print(f"Offset for img2: {offset2}")

        # Adjust the resampling origin to account for the offset (add the offset, not subtract)
        img1_resampled = resample_image(img1_cropped, new_origin + offset1, new_spacing, new_size, new_direction)
        img2_resampled = resample_image(img2_cropped, new_origin + offset2, new_spacing, new_size, new_direction)

        # === Convert resampled images to numpy arrays for visualization ===
        np_img1 = sitk.GetArrayFromImage(img1_resampled)
        np_img2 = sitk.GetArrayFromImage(img2_resampled)

        # Handle 3D volumes with a single slice
        if np_img1.ndim == 3 and np_img1.shape[0] == 1:
            np_img1 = np_img1[0]
        if np_img2.ndim == 3 and np_img2.shape[0] == 1:
            np_img2 = np_img2[0]

        #Registration
        np_img1_f32 = np_img1.astype(np.float32)
        np_img2_f32 = np_img2.astype(np.float32)

        # Create ants images and set spacing, origin
        ants_img1 = ants.from_numpy(np_img1_f32)
        ants_img1.set_spacing(img1_resampled.GetSpacing()[:-1])
        ants_img1.set_origin(img1_resampled.GetOrigin()[:-1])

        ants_img2 = ants.from_numpy(np_img2_f32)
        ants_img2.set_spacing(img2_resampled.GetSpacing()[:-1])
        ants_img2.set_origin(img2_resampled.GetOrigin()[:-1])


        # Extract warped output
        np_img2 = ants.registration(fixed = ants_img1 , moving = ants_img2, type_of_transform ='SyNAggro')['warpedmovout']
        np_img2 = np_img2.numpy()

        # === Visualize original and resampled images ===
        if not isinstance(mask, sitk.Image):
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)  # Shape: (1, H, W)
            mask = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask.CopyInformation(img1)  # Align with Image 1, not Image 2

        # === Crop the mask using the same overlap bounds ===
        mask_cropped, mask_cropped_origin = crop_to_overlap(mask, overlap_min, overlap_max)


        # === Compute the same offset as img2 ===
        mask_offset = mask_cropped_origin - new_origin
        # print(f"Offset for mask: {mask_offset}")

        # === Resample the mask using nearest neighbor interpolation ===
        def resample_mask(image, new_origin, new_spacing, new_size, new_direction):
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputOrigin(new_origin)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size.tolist())
            resampler.SetOutputDirection(new_direction)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            return resampler.Execute(image)

        mask_resampled = resample_mask(mask_cropped, new_origin + mask_offset, new_spacing, new_size, new_direction)


        # === Convert to numpy array if needed for visualization ===
        np_mask = sitk.GetArrayFromImage(mask_resampled)
        if np_mask.ndim == 3 and np_mask.shape[0] == 1:
            np_mask = np_mask[0]


        # If single-slice, extract the 2D arrays
        if np_img2.ndim == 3 and np_img2.shape[0] == 1:
            np_img2 = np_img2[0]
        if np_mask.ndim == 3 and np_mask.shape[0] == 1:
            np_mask = np_mask[0]

        # Ensure mask is boolean
        binary_mask = np_mask > 0

        # Get the pixel values in img2 where mask is present
        masked_values = np_img2[binary_mask]

        # Compute the mean intensity
        if masked_values.size == 0:
            print("Warning: No mask-covered region found in img2.")
            mean_intensity = np.nan
        else:
            mean_intensity = masked_values.mean()
        if mean_intensity > 30 and mean_intensity < 65:
            ave_HU.append(mean_intensity)

    print(f"Average intensity in img2 under transformed mask: {np.mean(ave_HU[:-5])}")

    ##Visulization

    # plt.figure(figsize=(12, 8))

    # plt.subplot(2, 2, 1)
    # plt.imshow(sitk.GetArrayViewFromImage(img1)[0], cmap='gray')
    # plt.imshow(eroded_mask, cmap='spring', alpha=0.4)
    # plt.title('Original Image 1')
    # plt.grid(True, color='r', linestyle='--', linewidth=0.5)  # red dashed grid

    # plt.axis('on')  # show axes

    # plt.subplot(2, 2, 2)
    # plt.imshow(sitk.GetArrayViewFromImage(img2)[0], cmap='gray')
    # plt.imshow(eroded_mask, cmap='spring', alpha=0.4)
    # plt.title('Original Image 2')
    # plt.grid(True, color='r', linestyle='--', linewidth=0.5)
    # plt.axis('on')

    # plt.subplot(2, 2, 3)
    # plt.imshow(np_img1, cmap='gray')
    # plt.title('Resampled Image 1 (Overlap)')
    # plt.grid(True, color='r', linestyle='--', linewidth=0.5)
    # plt.imshow(np_mask, cmap='spring', alpha=0.4)
    # plt.axis('on')

    # plt.subplot(2, 2, 4)
    # plt.imshow(np_img2, cmap='gray')
    # plt.title('Resampled Image 2 (Overlap)')
    # plt.grid(True, color='r', linestyle='--', linewidth=0.5)
    # plt.imshow(np_mask, cmap='spring', alpha=0.4)
    # plt.axis('on')

    # plt.tight_layout()
    # plt.show()


    return(ave_HU)
