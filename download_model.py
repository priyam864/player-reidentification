"""
Utility script to download the YOLO model for player detection.
"""

import os
import requests
from tqdm import tqdm


def download_file(url: str, destination: str) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Successfully downloaded {destination}")
        return True
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def download_model():
    """
    Download the player detection model.
    """
    print("Downloading YOLOv11 player detection model...")
    
    # Google Drive file ID from the provided link
    file_id = "1-5fOSHOSB9UXYPenOoZNAMScrePVCMD"
    
    # Create download URL for Google Drive
    url = f"https://drive.usercontent.google.com/download?id=1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD&export=download&authuser=0"
    
    # Destination path
    destination = "models/best.pt"
    
    
    print(f"Downloading from Google Drive...")
    print(f"Destination: {destination}")
    
    try:
        # Use requests to handle Google Drive download
        import requests
        import shutil
        
        session = requests.Session()
        
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Check if it's a redirect to download confirmation
        if "text/html" in response.headers.get("content-type", ""):
            # Extract the download URL from the confirmation page
            from urllib.parse import parse_qs, urlparse
            
            # Try direct download link format
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = session.get(direct_url, stream=True)
            response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download the file
        with open(destination, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        print(f"✅ Model downloaded successfully!")
        print(f"Model saved to: {destination}")
        
        # Verify file size
        file_size = os.path.getsize(destination)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        print("\n📥 Manual Download Instructions:")
        print("1. Go to: https://drive.google.com/file/d/1-5fOSHOSB9UXYPenOoZNAMScrePVCMD/view")
        print("2. Click 'Download' button")
        print(f"3. Save the file as: {destination}")
        print("\nNote: This is a fine-tuned YOLOv11 model for player detection")
        return False


if __name__ == "__main__":
    download_model()
