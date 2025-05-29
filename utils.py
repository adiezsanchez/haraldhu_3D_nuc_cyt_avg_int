import nd2
import czifile
from pathlib import Path
import pyclesperanto_prototype as cle

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

        print(f"Pixel size: {pixel_size_x:.3f} µm x {pixel_size_y:.3f} µm")
        print(f"Voxel (Z-step) size: {voxel_size_z:.3f} µm")

    return pixel_size_x, pixel_size_y, voxel_size_z