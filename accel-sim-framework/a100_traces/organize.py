#!/usr/bin/env python3
import os
import subprocess

# List of i values (files including name _{i}_kernelslist.g)
i_list = [256, 320, 512, 1024, 1280, 2048, 4096, 5120, 8192]

# ---- Part0: Extract all .tar.gz files in current directory ----
current_dir = os.getcwd()

for file_name in os.listdir(current_dir):
    if file_name.endswith(".tar.gz"):
        file_path = os.path.join(current_dir, file_name)
        
        # Run tar -xvf <file>
        print(f"Extracting: {file_name}")
        subprocess.run(["tar", "-xvf", file_path], check=True)

# ---- Part1: draw .traceg.xz, kernelslist.g files to current directory ----

# Get current working directory
current_dir = os.getcwd()

# Loop through all items in current directory
for item in os.listdir(current_dir):
    folder_path = os.path.join(current_dir, item)

    # Check if this item is a folder
    if os.path.isdir(folder_path):
        
        # Loop through all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(file_path):

                # Check if filename contains "traceg"
                if "traceg" in file_name:
                    target_path = os.path.join(current_dir, file_name)
                    print(f"Moving {file_path} to {target_path}")
                    os.rename(file_path, target_path)

                # Check for exact match kernelslist.g
                elif file_name == "kernelslist.g":
                    new_filename = f"{item}_kernelslist.g"
                    target_path = os.path.join(current_dir, new_filename)
                    print(f"Renaming and moving {file_path} to {target_path}")
                    os.rename(file_path, target_path)


# ---- Part2: Create new kernelslist.g ----
# 1. Create (or overwrite) kernelslist.g
output_path = os.path.join(current_dir, "kernelslist.g")

# Overwrite if file exists
with open(output_path, "w") as outfile:
    pass

# 3. Get list of files in current dir
files_in_dir = os.listdir(current_dir)

# 4. Loop over i values
for i in i_list:
    search_suffix = f"_{i}_kernelslist.g"

    # Find all files ending with _{i}_kernelslist.g
    matching_files = [
        f for f in files_in_dir
        if f.endswith(search_suffix)
    ]

    for match_file in matching_files:
        match_path = os.path.join(current_dir, match_file)

        with open(match_path, "r") as f:
            for line in f:
                line = line.strip()

                # Check if this line matches any file name in the directory
                if line in files_in_dir:
                    with open(output_path, "a") as outfile:
                        outfile.write(line + "\n")
                        print(f"Added line to kernelslist.g: {line}")
