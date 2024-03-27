import os
import sys
import re

"""
Example usage: 
```python temp_rename.py <folder_path> 2```

For every file in <folder_path>, get the number in the filename, and increment it by 2, eg. `model-0.log` -> `model-2.log`

"""

def rename_files(folder_path, number):
    # Check if the folder path exists
    if not os.path.isdir(folder_path):
        print("Error: Folder does not exist.")
        return
    
    # first pass: add underscore to all files
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        new_filename = "_" + filename
        os.rename(filepath, os.path.join(folder_path, new_filename))
        print(f"Renamed '{filename}' to '{new_filename}'")

    # second pass: increment number and remove underscore from all files
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Check if it's a file
        if os.path.isfile(filepath):
            match = re.search(r'\d+', filename)

            if match:
                extracted_number = match.group()
                new_filename = filename.replace(extracted_number, str(int(extracted_number) + int(number)))
                new_filename = new_filename[1:]
                os.rename(filepath, os.path.join(folder_path, new_filename))
                print(f"Renamed '{filename}' to '{new_filename}'")

if __name__ == "__main__":
    # Check if two command line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <folder_path> <number>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    number = sys.argv[2]
    
    # Call the function
    rename_files(folder_path, number)
