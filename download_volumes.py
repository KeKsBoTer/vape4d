# %%
import numpy as np
import nibabel as nib

# %%

from matplotlib.colors import ListedColormap


def parse_sav_file(filepath):
    """
    Parses the content of a BluntFin .txt file.

    Args:
        filepath (str): The path to the BluntFin .txt file.

    Returns:
        dict: A dictionary containing the parsed data, or None if the file
              cannot be read or parsed.
    """
    parsed_data = {}
    current_section = None
    tf_points_buffer = {} # Buffer to collect re, ge, be, ra, ga, ba for TF section
    tf_point_keys = ['re', 'ge', 'be', 'ra', 'ga', 'ba']
    tf_point_key_index = 0

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines and comments
                    continue

                # Check for section headers
                if line.endswith(':'):
                    section_name = line[:-1]
                    parsed_data[section_name] = {}
                    current_section = section_name
                    # Initialize list for TF points if it's the TF section
                    if current_section == 'TF':
                        parsed_data[current_section]['points'] = []
                    tf_point_key_index = 0 # Reset index for new section
                    continue

                # Parse key-value pairs within sections
                if current_section:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Attempt to convert value to float or int if possible
                        try:
                            if '.' in value or 'e' in value.lower():
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass # Keep as string if conversion fails

                        if current_section == 'TF' and key in tf_point_keys:
                            # Collect values for TF points
                            tf_points_buffer[key] = value
                            tf_point_key_index += 1
                            if tf_point_key_index == len(tf_point_keys):
                                # Collected all keys for a point, add to list
                                parsed_data[current_section]['points'].append(tf_points_buffer)
                                tf_points_buffer = {} # Reset buffer
                                tf_point_key_index = 0 # Reset index
                        else:
                            # Store regular key-value pairs
                            parsed_data[current_section][key] = value
                            # If we are in TF section but parsing initial keys (like res, rescale),
                            # ensure we reset the point index in case these come after some points
                            # This handles the specific structure observed where res comes first.
                            if current_section == 'TF' and key not in tf_point_keys:
                                tf_point_key_index = 0


    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return None

    return parsed_data

# %%
import re

def parse_pvm_info(output_text):
    """
    Parses the PVM info output text and extracts relevant details.

    Args:
        output_text (str): The string containing the terminal output.

    Returns:
        dict: A dictionary containing the extracted information, or None if parsing fails.
    """
    data = {}

    # Regex to capture the main volume dimensions
    volume_match = re.search(r"found volume with width=(\d+) height=(\d+) depth=(\d+) components=(\d+)", output_text)
    if volume_match:
        data['width'] = int(volume_match.group(1))
        data['height'] = int(volume_match.group(2))
        data['depth'] = int(volume_match.group(3))
        data['components'] = int(volume_match.group(4))

    # Regex to capture edge length (optional)
    scale_match = re.search(r"and edge length (\S+)/(\S+)/(\S+)", output_text)
    if scale_match:
        data['scalex'] = float(scale_match.group(1))
        data['scaley'] = float(scale_match.group(2))
        data['scalez'] = float(scale_match.group(3))

    # Regex to capture data checksum
    checksum_match = re.search(r"and data checksum=(\S+)", output_text)
    if checksum_match:
        data['checksum'] = checksum_match.group(1)

    # Regex to capture multiline fields
    # This uses a non-greedy match (.*?) for the content between the field header and the next field header or the end of the string
    description_match = re.search(r"object description:\n(.*?)(?=\n(?:courtesy information:|scan parameters:|additonal comments:|$))", output_text, re.DOTALL)
    if description_match:
        data['description'] = description_match.group(1).strip()

    courtesy_match = re.search(r"courtesy information:\n(.*?)(?=\n(?:scan parameters:|additonal comments:|$))", output_text, re.DOTALL)
    if courtesy_match:
        data['courtesy'] = courtesy_match.group(1).strip()

    parameters_match = re.search(r"scan parameters:\n(.*?)(?=\n(?:additonal comments:|$))", output_text, re.DOTALL)
    if parameters_match:
        data['parameters'] = parameters_match.group(1).strip()

    comment_match = re.search(r"additonal comments:\n(.*)", output_text, re.DOTALL)
    if comment_match:
        data['comment'] = comment_match.group(1).strip()

    return data if data else None

# %%
import glob
import subprocess
import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
from colormaps import export_colormap

PVMINFO_BINARY = "/home/niedermayr/Downloads/VIEWER-5.2/viewer/tools/pvminfo"
PVM2RAW_BINARY = "/home/niedermayr/Downloads/VIEWER-5.2/viewer/tools/pvm2raw"

def download_files_from_url(url, extensions, download_dir="downloaded_files"):
    """
    Downloads files with specified extensions from a given URL.

    Args:
        url (str): The URL of the webpage to scrape.
        extensions (list): A list of file extensions to look for (e.g., ['.pvm', '.sav']).
        download_dir (str): The directory to save the downloaded files.
    """
    # Create download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created directory: {download_dir}")

    try:
        # Fetch the webpage content
        print(f"Fetching page: {url}")
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all anchor tags (<a>) which contain links
        links = soup.find_all('a')

        found_files = []

        # Iterate through the links
        for link in links:
            href = link.get('href')
            if href:
                # Construct absolute URL to handle relative links
                absolute_url = urllib.parse.urljoin(url, href)

                # Check if the URL ends with any of the target extensions (case-insensitive)
                if any(absolute_url.lower().endswith(ext.lower()) for ext in extensions):
                    found_files.append(absolute_url)

        if not found_files:
            print("No files with specified extensions found on the page.")
            return

        print(f"Found {len(found_files)} files to download.")

        # Download each found file
        for file_url in found_files:
            try:
                # Extract filename from the URL
                filename = os.path.basename(urllib.parse.urlparse(file_url).path)
                save_path = os.path.join(download_dir, filename)

                # Check if file already exists to avoid re-downloading
                # if os.path.exists(save_path):
                #     print(f"File already exists, skipping: {filename}")
                #     continue

                print(f"Downloading: {filename} from {file_url}")
                # Use stream=True for potentially large files
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status() # Check for bad response status
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Successfully downloaded: {filename}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_url}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {file_url}: {e}")
            
            
            
            if file_url.endswith('.sav'):
                try:
                    parsed_data = parse_sav_file(save_path)
                    cmap_name = filename.split(".")[0]+"_cmap"
                    print(f"Parsed .sav file: {cmap_name}")

                    tf = np.array([[p["re"],p["ge"],p["be"],p["ra"]] for p in parsed_data["TF"]["points"]])
                    cmap = ListedColormap(tf,cmap_name)
                    export_colormap(download_dir,cmap_name,cmap)
                except Exception as e:  
                    print(f"Error parsing {save_path} file: {e}")
                
                continue
            
            try:
                pvm_info_text = subprocess.run([PVMINFO_BINARY, save_path], capture_output=True, text=True).stdout.encode('utf-8')
                pvm_info = parse_pvm_info(pvm_info_text.decode('utf-8'))
                # print(f"Parsed PVM info for {filename}:", pvm_info)
            except subprocess.CalledProcessError as e:
                print(f"Error running pvminfo on {filename}: {e}")


            try:
                raw_filename = os.path.splitext(filename)[0] + ".raw"
                raw_save_path = os.path.join(download_dir, raw_filename)
                # print(f"Converting {filename} to {raw_filename} using pvm2raw")
                subprocess.run([PVM2RAW_BINARY, save_path, raw_save_path], check=True, capture_output=True)
                raw_file_path = glob.glob(os.path.join(download_dir, os.path.splitext(filename)[0]+"*.raw"))[0]
                # print(f"Successfully converted to {raw_file_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error running pvm2raw on {filename}: {e}")

            try:
                dtype = np.uint8 if pvm_info['components'] == 1 else np.uint16
                data = np.fromfile(raw_file_path,dtype=dtype).reshape(pvm_info['depth'], pvm_info['height'], pvm_info['width'])
                
                header = nib.Nifti1Header()
                header.set_data_shape(data.shape)
                header.set_data_dtype(dtype)
                affine = np.array([[pvm_info.get("scalex",1), 0, 0, 0],
                                    [0, pvm_info.get("scaley",1), 0, 0],
                                    [0, 0, pvm_info.get("scalez",1), 0],
                                    [0, 0, 0, 1]])
                
                out_nifiti = os.path.join(download_dir, os.path.splitext(filename)[0] + ".nii")
                nib.save(nib.nifti1.Nifti1Image(data,affine,header=header),out_nifiti)
                print(f"Successfully converted to NIfTI format: {out_nifiti}")
            except Exception as e:
                print(f"Error converting raw file to NIfTI: {e}")
                
                
                            

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

target_url = "http://volume.open-terrain.org/"
target_extensions = ['.pvm', '.sav']
download_directory = "downloaded_files"

download_files_from_url(target_url, target_extensions, download_directory)

print("\nDownload process finished.")

# %%



