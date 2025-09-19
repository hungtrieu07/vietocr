import os
import gdown
import yaml
import numpy as np
import uuid
import requests
import tempfile
import time
import shutil
from tqdm import tqdm
import zipfile

# Global cache for downloaded files to prevent re-downloading
_downloaded_files_cache = {}


def download_weights(uri, cached=None, md5=None, quiet=False):
    if uri.startswith("http"):
        return download(url=uri, quiet=quiet)
    return uri

import os
import shutil
import requests
from tqdm import tqdm
import gdown

def download(url, quiet=False):
    tmp_dir = "vietocr/downloaded_models"
    filename = url.split("/")[-1]
    full_path = os.path.join(tmp_dir, filename)

    os.makedirs(tmp_dir, exist_ok=True)

    # Check if any .pth already exists in tmp_dir
    existing_pth = [f for f in os.listdir(tmp_dir) if f.endswith(".pth")]
    if existing_pth:
        print(f".pth file already exists in {tmp_dir}: {existing_pth[0]}")
        return os.path.join(tmp_dir, existing_pth[0])

    # Check if this exact file already exists
    if os.path.exists(full_path):
        print(f"Model weight {full_path} exists. Ignore download!")
        return full_path

    # Google Drive
    if "drive.google.com" in url or "docs.google.com" in url:
        output_path = gdown.download(url, fuzzy=True, quiet=quiet)

        if output_path is None:
            raise RuntimeError(f"Failed to download from Google Drive: {url}")

        if output_path.endswith(".pth"):
            new_path = os.path.join(tmp_dir, os.path.basename(output_path))
            shutil.move(output_path, new_path)
            return new_path

        return output_path

    # Normal HTTP
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(full_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), disable=quiet):
                if chunk:
                    f.write(chunk)
    return full_path



def download_config(id):
    # Create config directory if it doesn't exist
    config_dir = "vietocr/config"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    # Check if config file already exists locally
    config_file_path = os.path.join(config_dir, id)
    if os.path.exists(config_file_path):
        print("Config file {} exists. Loading from local file.".format(config_file_path))
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    # Mapping of config filenames to their Google Drive file IDs
    # You'll need to get these individual file IDs from the Google Drive folder
    config_file_mapping = {
        "base.yml": "1PM0de_becZp9oYPN-MFYby4ikOYKQJ1i",  
        "resnet_fpn_transformer.yml": "14rkiwa5RojEgTU0D-BUSnM8u46zn4QOX",
        "resnet-transformer.yml": "1GG-WgC09jbSqslp_9u0Eu0Zoerc0ET8-",
        "vgg-convseq2seq.yml": "1xvsTk2WCKYU1af-IhmqDnJL_gEpJkzhC",
        "vgg-seq2seq.yml": "1e9Ypb_U4XUpC2Q9d71Ymosm-I6m7GNGF",
        "vgg-transformer.yml": "1lUjyYR8DXWCX11Yeq0B6xq2AFCCJkND5",  
    }
    
    # Check if we have the file ID for this config
    if id not in config_file_mapping:
        print(f"No Google Drive file ID found for config: {id}")
        return None
    
    file_id = config_file_mapping[id]
    gdrive_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    
    print(f"Config file {id} not found locally. Downloading from Google Drive...")
    
    try:
        # Download the specific config file
        output_path = gdown.download(gdrive_url, fuzzy=True)
        
        if output_path is None:
            raise RuntimeError(f"Failed to download config file {id} from Google Drive")
        
        # Check if download was successful and not a .part file
        if output_path.endswith('.part'):
            print(f"Download incomplete (.part file detected): {output_path}")
            # Remove the .part file and try again
            try:
                os.remove(output_path)
            except:
                pass
            # Try downloading again
            output_path = gdown.download(gdrive_url, fuzzy=True)
            
            if output_path is None or output_path.endswith('.part'):
                raise RuntimeError(f"Download still incomplete after retry for {id}")
        
        print(f"Downloaded config file to: {output_path}")
        
        # Copy the downloaded file to our config directory with the correct name
        shutil.copy2(output_path, config_file_path)
        print(f"Config file {id} downloaded and saved to {config_file_path}")
        
        # Remove the original downloaded file
        try:
            os.remove(output_path)
        except:
            pass
        
        # Load and return the config
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
            
    except Exception as e:
        print(f"Error downloading config {id} from Google Drive: {e}")
        return None


def compute_accuracy(ground_truth, predictions, mode="full_sequence"):
    """
    Computes accuracy
    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :param mode: if 'per_char' is selected then
                 single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
                 if 'full_sequence' is selected then
                 single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
    :return: avg_label_accuracy
    """
    if mode == "per_char":

        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            total_count = len(label)
            correct_count = 0
            try:
                for i, tmp in enumerate(label):
                    if tmp == prediction[i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(prediction) == 0:
                        accuracy.append(1)
                    else:
                        accuracy.append(0)
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    elif mode == "full_sequence":
        try:
            correct_count = 0
            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                if prediction == label:
                    correct_count += 1
            avg_accuracy = correct_count / len(ground_truth)
        except ZeroDivisionError:
            if not predictions:
                avg_accuracy = 1
            else:
                avg_accuracy = 0
    else:
        raise NotImplementedError(
            "Other accuracy compute mode has not been implemented"
        )

    return avg_accuracy
