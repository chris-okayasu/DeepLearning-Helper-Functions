import zipfile
import os
from pathlib import Path

def fun_unzip_data(filename, folder_name="dataset", extract_path=None, delete_zip=False):
    """
    Unzips a zip file into a specified directory. By default, it extracts to '/content/{folder_name}' in Google Colab.

    Args:
        filename (str): Name of the zip file (must be in the current working directory).
        folder_name (str, optional): Default name of the folder where files will be extracted. 
                                     Defaults to "dataset".
        extract_path (str, optional): Custom path to extract files. If None, defaults to '/content/{folder_name}'.
        delete_zip (bool, optional): Whether to delete the zip file after extraction. 
                                     Defaults to False.
    """

    # Determine the destination directory
    if extract_path:
        dest_dir = Path(extract_path).resolve()  # Use the custom path
    else:
        dest_dir = Path("/content") / folder_name  # Default to Colab's /content/

    dest_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Full path to the zip file
    zip_path = Path.cwd() / filename 

    # Check if the file exists before attempting extraction
    if not zip_path.exists():
        print(f"❌ Error: File '{filename}' not found in {Path.cwd()}")
        return

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = zip_ref.namelist()  # List of files inside the zip
            zip_ref.extractall(dest_dir)

        print(f"✅ Extracted {len(file_list)} files from '{filename}' to '{dest_dir}'")

        # Optionally delete the zip file after extraction
        if delete_zip:
            zip_path.unlink()
            print(f"🗑️ Deleted '{filename}' after extraction")

    except zipfile.BadZipFile:
        print(f"❌ Error: '{filename}' is not a valid zip file or is corrupted.")
