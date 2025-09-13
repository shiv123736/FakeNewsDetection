# data_download.py
import os
import webbrowser
import shutil

def copy_files_if_exists(source_dir, dest_dir):
    """Copy CSV files if they exist in the source directory."""
    fake_source = os.path.join(source_dir, "Fake.csv")
    true_source = os.path.join(source_dir, "True.csv")
    fake_dest = os.path.join(dest_dir, "Fake.csv")
    true_dest = os.path.join(dest_dir, "True.csv")
    
    success = True
    
    # Try to copy Fake.csv if it exists
    if os.path.exists(fake_source):
        print(f"Copying {fake_source} to {fake_dest}")
        try:
            shutil.copy2(fake_source, fake_dest)
            print("✅ Successfully copied Fake.csv")
        except Exception as e:
            print(f"❌ Failed to copy Fake.csv: {e}")
            success = False
    else:
        print(f"❌ Source file {fake_source} does not exist")
        success = False
    
    # Try to copy True.csv if it exists
    if os.path.exists(true_source):
        print(f"Copying {true_source} to {true_dest}")
        try:
            shutil.copy2(true_source, true_dest)
            print("✅ Successfully copied True.csv")
        except Exception as e:
            print(f"❌ Failed to copy True.csv: {e}")
            success = False
    else:
        print(f"❌ Source file {true_source} does not exist")
        success = False
    
    return success

def main():
    # Define directories
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # File paths in the data directory
    fake_csv_path = os.path.join(data_dir, "Fake.csv")
    true_csv_path = os.path.join(data_dir, "True.csv")
    
    # Check if files already exist
    if os.path.exists(fake_csv_path) and os.path.exists(true_csv_path):
        print(f"Dataset files already exist in {data_dir} directory.")
        return
    
    print("Fake News Dataset Setup")
    print("======================")
    print("\nThis script will help you set up the dataset for the fake news detection project.")
    
    # Option 1: Check for the files in a specific folder
    print("\nOption 1: Check for the files in a specific folder")
    #folder_path = input("Enter the folder path where Fake.csv and True.csv are located (or press Enter to skip): ")
    folder_path = "data"
    
    if folder_path and os.path.isdir(folder_path):
        print(f"\nChecking {folder_path} for CSV files...")
        if copy_files_if_exists(folder_path, data_dir):
            print("\nSuccess! Files have been copied to the data directory.")
            return
        else:
            print("\nCould not find or copy the required files from the specified folder.")
    else:
        if folder_path:
            print(f"\nThe folder '{folder_path}' doesn't exist or is not accessible.")
    
    # Option 2: Manual download
    print("\nOption 2: Manually download the files")
    print("1. Go to your Google Drive folder: https://drive.google.com/drive/folders/1PGp0U4v7Ymcvk40YdRPtUllhQTIabf-N")
    print("2. Download 'Fake.csv' and 'True.csv'")
    print("3. Place them in the 'data' directory of this project")
    
    # Open the folder in the web browser for convenience
    open_browser = input("\nWould you like to open the Google Drive folder in your web browser? (y/n): ").lower().startswith('y')
    if open_browser:
        folder_url = "https://drive.google.com/drive/folders/1PGp0U4v7Ymcvk40YdRPtUllhQTIabf-N?usp=drive_link"
        webbrowser.open(folder_url)
        
    input("\nPress Enter after you've downloaded the files and placed them in the 'data' directory...")
    
    # Check if files now exist in the data directory
    if os.path.exists(fake_csv_path) and os.path.exists(true_csv_path):
        print("\nSuccess! Files found in the data directory.")
        print("You can now proceed with preprocessing.")
    else:
        print("\nFiles not found in the data directory.")
        print("Please manually download 'Fake.csv' and 'True.csv' from Google Drive")
        print("and place them in the 'data' directory before proceeding.")

if __name__ == "__main__":
    main()