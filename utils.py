import nd2
import czifile
from pathlib import Path
import pyclesperanto_prototype as cle
import numpy as np
from skimage import measure

cle.select_device("RTX")

def list_images (directory_path, format=None):

    # Create an empty list to store all image filepaths within the dataset directory
    images = []

    # If manually defined format
    if format:
        for file_path in directory_path.glob(f"*.{format}"):
            images.append(str(file_path))

    else:
        # Iterate through the .czi and .nd2 files in the directory
        for file_path in directory_path.glob("*.czi"):
            images.append(str(file_path))
            
        for file_path in directory_path.glob("*.nd2"):
            images.append(str(file_path))

    return images

def read_image (image, slicing_factor_xy, slicing_factor_z):
    """Read raw image microscope files, apply downsampling if needed and return filename and a numpy array"""
    # Read path storing raw image and extract filename
    file_path = Path(image)
    filename = file_path.stem

    # Extract file extension
    extension = file_path.suffix

    # Read the image file (either .czi or .nd2)
    if extension == ".czi":
        # Stack from .czi (ch, z, x, y)
        img = czifile.imread(image)
        # Remove singleton dimensions
        img = img.squeeze()

    elif extension == ".nd2":
        # Stack from .nd2 (z, ch, x, y)
        img = nd2.imread(image)
        # Transpose to output (ch, z, x, y)
        img = img.transpose(1, 0, 2, 3)

    else:
        print ("Implement new file reader")

    print(f"\n\nImage analyzed: {filename}")
    print(f"Original Array shape: {img.shape}")

    # Apply slicing trick to reduce image size (xy resolution)
    try:
        img = img[:, ::slicing_factor_z, ::slicing_factor_xy, ::slicing_factor_xy]
    except IndexError as e:
        print(f"Slicing Error: {e}")
        print(f"Slicing parameters: Slicing_XY:{slicing_factor_xy} Slicing_Z:{slicing_factor_z} ")

    # Feedback for researcher
    print(f"Compressed Array shape: {img.shape}")

    return img, filename

def extract_pixel_size(image):

    with nd2.ND2File(image) as nd2_data:
        # Get the first channel's volume metadata
        first_channel = nd2_data.metadata.channels[0]
        voxel_size = first_channel.volume.axesCalibration  # X, Y, Z calibration

        # Extract pixel sizes (tuple unpacking)
        pixel_size_x, pixel_size_y, voxel_size_z = voxel_size

        #print(f"Pixel size: {pixel_size_x:.3f} µm x {pixel_size_y:.3f} µm")
        #print(f"Voxel (Z-step) size: {voxel_size_z:.3f} µm")

    return pixel_size_x, pixel_size_y, voxel_size_z

def correct_pixel_size(voxel_size, slicing_factor_xy, slicing_factor_z):
    # Correct pixel sizes upon downsampling (slicing_factor =! None)

    # Tuple unpacking
    pixel_size_x, pixel_size_y, voxel_size_z = voxel_size

    if slicing_factor_xy == None:
        slicing_factor_xy = 1

    if slicing_factor_z == None:
        slicing_factor_z = 1

    pixel_size_x = pixel_size_x * slicing_factor_xy
    pixel_size_y = pixel_size_y * slicing_factor_xy
    voxel_size_z = voxel_size_z * slicing_factor_z

    return pixel_size_x, pixel_size_y, voxel_size_z

def make_isotropic(image, scaling_x_um, scaling_y_um, scaling_z_um):
    """Scale the image with the voxel size used as scaling factor to get an image stack with isotropic voxels"""
    # Set the x and y scaling to 1 to avoid resizing (compressing the input image), this way the rescaling factor is only applied to z
    multiplier = 1 / scaling_x_um

    scaling_x_um = scaling_x_um * multiplier
    scaling_y_um = scaling_y_um * multiplier
    scaling_z_um = scaling_z_um * multiplier

    image_resampled = cle.scale(
        image,
        factor_x=scaling_x_um,
        factor_y=scaling_y_um,
        factor_z=scaling_z_um,
        auto_size=True,
    )

    return image_resampled

def remove_labels_touching_edge(labels):
    """
    Removes labels that are touching the edges of a 2D or 3D labeled array.

    Parameters:
    -----------
    labels : np.ndarray
        A 2D or 3D NumPy array of labeled structures (e.g. nuclei), where each unique integer value represents
        a different label and `0` represents the background.

    Returns:
    --------
    np.ndarray
        A 2D or 3D array with the labels that are touching the image border removed (set to 0).
    """
    labels = labels.copy()

    if labels.ndim == 2:
        contour_mask = np.zeros_like(labels, dtype=bool)
        contour_mask[0, :] = True  # Top edge
        contour_mask[-1, :] = True  # Bottom edge
        contour_mask[:, 0] = True  # Left edge
        contour_mask[:, -1] = True  # Right edge

    elif labels.ndim == 3:
        contour_mask = np.zeros_like(labels, dtype=bool)
        contour_mask[:, 0, :] = True   # Top Y
        contour_mask[:, -1, :] = True  # Bottom Y
        contour_mask[:, :, 0] = True   # Left X
        contour_mask[:, :, -1] = True  # Right X

    else:
        raise ValueError("Input labels must be a 2D or 3D array.")

    # Identify labels that touch the edge
    intersecting_labels = np.unique(labels[contour_mask])
    intersecting_labels = intersecting_labels[intersecting_labels != 0]

    # Remove those labels
    np.putmask(labels, np.isin(labels, intersecting_labels), 0)

    # Relabel to ensure continuous labeling
    labels = measure.label(labels, connectivity=1)

    return labels

def simulate_cytoplasm(nuclei_labels, dilation_radius=2, erosion_radius=0):

    if erosion_radius >= 1:

        # Erode nuclei_labels to maintain a closed cytoplasmic region when labels are touching (if needed)
        eroded_nuclei_labels = cle.erode_labels(nuclei_labels, radius=erosion_radius)
        eroded_nuclei_labels = cle.pull(eroded_nuclei_labels)
        nuclei_labels = eroded_nuclei_labels

    # Dilate nuclei labels to simulate the surrounding cytoplasm
    cyto_nuclei_labels = cle.dilate_labels(nuclei_labels, radius=dilation_radius)
    cytoplasm = cle.pull(cyto_nuclei_labels)

    # Create a binary mask of the nuclei
    nuclei_mask = nuclei_labels > 0

    # Set the corresponding values in the cyto_nuclei_labels array to zero
    cytoplasm[nuclei_mask] = 0

    return cytoplasm