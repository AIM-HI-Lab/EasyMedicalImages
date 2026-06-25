import sys
import time
import json
import tempfile
from shutil import copy
from pathlib import Path

import pydicom
from tqdm import tqdm
import SimpleITK as sitk


METADATA_TO_PULL = {
    # More dates/times than we could ever need
    "Acquisition DateTime": (0x0008, 0x002A),
    "Acquisition Date": (0x0008, 0x0022),
    "Acquisition Time": (0x0008, 0x0032),
    "Content Date": (0x0008, 0x0023),
    "Content Time": (0x0008, 0x0033),
    "Study Date": (0x0008, 0x0020),
    "Study Time": (0x0008, 0x0030),
    "Series Date": (0x0008, 0x0021),
    "Series Time": (0x0008, 0x0031),
    "Instance Creation Date": (0x0008, 0x0012),
    "Instance Creation Time": (0x0008, 0x0013),
    "Image Date": (0x0008, 0x0024),
    "Image Time": (0x0008, 0x0034),
    "Acquisition DateTime": (0x0008, 0x002A),

    # Everything else
    "Series Instance UID": (0x0020, 0x000E),
    "Manufacturer": (0x0008, 0x0070),
    "Institution Name": (0x0008, 0x0080),
    "Patient ID": (0x0010, 0x0020),
    "Body Part Examined": (0x0018, 0x0015),
    "Series Description": (0x0008, 0x103E),
    "Slice Thickness": (0x0018, 0x0050),
    "KVP": (0x0018, 0x0060),
    "Data Collection Diameter": (0x0018, 0x0090),
    "Device Serial Number": (0x0018, 0x1000),
    "Reconstruction Diameter": (0x0018, 0x1100),
    "Protocol Name": (0x0018, 0x1030),
    "Distance Source to Detector": (0x0018, 0x1110),
    "Distance Source to Patient": (0x0018, 0x1111),
    "Gantry/Detector Tilt": (0x0018, 0x1120),
    "Table Height": (0x0018, 0x1130),
    "Exposure Time": (0x0018, 0x1150),
    "X-Ray Tube Current": (0x0018, 0x1151),
    "Exposure": (0x0018, 0x1152),
    "Filter Type": (0x0018, 0x1160),
    "Generator Power": (0x0018, 0x1170),
    "Convolution Kernel": (0x0018, 0x1210),
    "Rows": (0x0028, 0x0010),
    "Columns": (0x0028, 0x0011),
    "Pixel Spacing": (0x0028, 0x0030)
}


def sanitize_metadata_key(key: str):
    return key.replace(" ", "_")\
              .replace("/", "_")\
              .replace("(", "")\
              .replace(")", "").lower()


def is_dcm(pth: Path):
    # First, check whether the path has a .dcm extension
    if pth.suffix == ".dcm":
        return True

    # If no extenstion, check whether the file is a DICOM file
    try:
        # Attempt to read the file as a DICOM file
        ds = pydicom.dcmread(str(pth), force=True)
        # Check for the presence of a DICOM tag (e.g., 'PatientID')
        return 'PatientID' in ds
    except pydicom.errors.InvalidDicomError:
        return False


def find_dcm_pths(pth: Path, metadata: dict = None):
    # Iterate over all items in the path
    for item in pth.iterdir():
        # If the item is a directory, recursively search for DICOM files
        if item.is_dir():
            metadata = find_dcm_pths(item, metadata)
            continue

        # If the item is not a file, skip it
        elif not item.is_file():
            continue

        # If the item is a file, check if it has already been considered
        if str(item) in metadata["files_considered"]:
            continue

        # If not considered, check if it is a DICOM file
        if is_dcm(item):
            metadata["dcm_pths"].append(item)

        # Regardless, add the path to the list of considered paths
        metadata["files_considered"].append(str(item))

    return metadata


def get_series_metadata(dcm_pths):
    # Initialize an empty dictionary to store the metadata
    n_slices = len(dcm_pths)
    metadata = {
        "n_slices": n_slices,
    }

    # Iterate over all DICOM files in the series
    for dcm_pth in dcm_pths:
        # If all of the metadata has been found, break the loop
        if not any([
            key not in metadata
            for key in [
                sanitize_metadata_key(key)
                for key in METADATA_TO_PULL.keys()
            ]
        ]):
            break

        # Read the DICOM file
        ds = pydicom.dcmread(str(dcm_pth), force=True)

        # Extract the metadata from the DICOM file
        for key, tag in METADATA_TO_PULL.items():
            san_key = sanitize_metadata_key(key)
            if san_key not in metadata:
                try:
                    metadata[san_key] = ds[tag].value
                except KeyError:
                    metadata[san_key] = None

    # Return the metadata as a dictionary
    return metadata


def get_unique_series_id(dcm_pth: Path):
    # Read the DICOM file
    ds = pydicom.dcmread(str(dcm_pth), force=True)

    # Extract the SeriesInstanceUID
    sid = ds.SeriesInstanceUID

    # Return the SeriesInstanceUID as a string
    return str(sid)


def dcm_list_to_nii(dcm_pths, nii_pth):
    # Create a temporary directory to copy these images
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_pth = Path(temp_dir.name)
    temp_dir_pth.mkdir(parents=True, exist_ok=True)

    # Copy all DICOM files to the temporary directory
    for dcm_pth in dcm_pths:
        dcm_name = Path(dcm_pth).name
        dcm_dst_pth = temp_dir_pth / dcm_name
        copy(str(dcm_pth), str(dcm_dst_pth))

    # Convert the DICOM files to a NIfTI file using sitk
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(temp_dir_pth))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Check how many slices/voxels there are in each direction
    size_x, size_y, size_z = image.GetSize()
    if size_x < 5 or size_y < 5 or size_z < 5:
        # Too few slices in at least one direction → skip writing
        # (the TemporaryDirectory will auto‐clean on function exit)
        return False

    sitk.WriteImage(image, str(nii_pth))
    return True


def convert(src_pth: Path, dst_pth: Path):
    # Ensure that the source and destination paths are Path objects
    src_pth = Path(src_pth)
    dst_pth = Path(dst_pth)

    # Ensure the source path exists
    if not src_pth.exists():
        raise FileNotFoundError(f"Source path {src_pth} does not exist.")
    
    # Ensure the destination path exists
    dst_pth.mkdir(parents=True, exist_ok=True)

    # If the destination path has a metadata file, load it
    metadata_pth = dst_pth / "metadata.json"
    if metadata_pth.exists():
        with open(metadata_pth, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Ensure a list of all dcm paths is included in metadata
    if "dcm_pths" not in metadata:
        metadata["dcm_pths"] = []
    if "files_considered" not in metadata:
        metadata["files_considered"] = []

    # Recursively locate all DICOM files in the source path
    print("Step 1 of 4: Locating all DICOM files...")
    metadata = find_dcm_pths(src_pth, metadata)

    # Ensure that we have a dict linking paths to series
    if "file_to_series" not in metadata:
        metadata["file_to_series"] = {}

    # Initialize a set of series IDs
    if "series_to_files" not in metadata:
        metadata["series_to_files"] = {}

    # Initialize a queue
    pth_queue = [
        x for x in metadata["dcm_pths"]
        if str(x) not in metadata["file_to_series"]
    ]
    print(
        "Found {} DICOM files in folder.".format(
            len(metadata["dcm_pths"])
        )
    )
    print(f"Processing {len(pth_queue)} must be sorted.")

    # Sort the DICOM files according to their series
    print("Step 2 of 4: Sorting DICOM files by series...")
    split_time = time.time()
    for dcm_pth in tqdm(pth_queue, file=sys.stdout):
        # Update the metadata with the series ID
        sid = get_unique_series_id(dcm_pth)
        metadata["file_to_series"][str(dcm_pth)] = sid
        if sid not in metadata["series_to_files"]:
            metadata["series_to_files"][sid] = []
        metadata["series_to_files"][sid].append(str(dcm_pth))

        # If more than 60 seconds have passed, save the metadata
        if time.time() - split_time > 60:
            with open(metadata_pth, "w") as f:
                json.dump(metadata, f, default=str, indent=2)
            split_time = time.time()

    # Save the metadata
    with open(metadata_pth, "w") as f:
        json.dump(metadata, f, default=str, indent=2)

    # Print the number of series found
    n_series = len(metadata["series_to_files"].keys())
    print(f"Found {n_series} unique series.")

    # Compute metadata for each series
    if "series_metadata" not in metadata:
        metadata["series_metadata"] = {}
    print("Step 3 of 4: Pulling metadata for each series...")
    smd_queue = []
    for sid, dcm_pths in metadata["series_to_files"].items():
        if sid not in metadata["series_metadata"]:
            smd_queue.append(sid)
    split_time = time.time()
    for sid in tqdm(smd_queue, file=sys.stdout):
        dcm_pths = metadata["series_to_files"][sid]
        smd = get_series_metadata(dcm_pths)
        metadata["series_metadata"][sid] = smd

        # If more than 60 seconds have passed, save the metadata
        if time.time() - split_time > 60:
            with open(metadata_pth, "w") as f:
                json.dump(metadata, f, default=str, indent=2)
            split_time = time.time()

    # Save the metadata
    with open(metadata_pth, "w") as f:
        json.dump(metadata, f, default=str, indent=2)

    # Initialize the conversion status in the metadata
    if "conversion_status" not in metadata:
        metadata["conversion_status"] = {}

    # Convert each series to NIfTI format
    print("Step 4 of 4: Converting DICOM files to NIfTI format...")
    split_time = time.time()
    for sid, dcm_pths in tqdm(
        metadata["series_to_files"].items(), file=sys.stdout
    ):
        # Determine the destination path for the series
        series_dst_pth = dst_pth / f"{sid}.nii.gz"

        # If the series has already been converted, skip it
        if series_dst_pth.exists():
            continue

        # Convert the series to NIfTI format
        try:
            dcm_list_to_nii(dcm_pths, series_dst_pth)
            metadata["conversion_status"][sid] = "success"
        except Exception as e:
            print(f"Failed to convert series {sid}: {e}")
            metadata["conversion_status"][sid] = "failed: " + str(e)

        # Save the metadata if it's been more than a minute
        if time.time() - split_time > 60:
            with open(metadata_pth, "w") as f:
                json.dump(metadata, f, default=str, indent=2)
            split_time = time.time()

    # Save the metadata
    with open(metadata_pth, "w") as f:
        json.dump(metadata, f, default=str, indent=2)