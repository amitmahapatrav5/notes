import os
import subprocess

def convert_md_to_ipynb(source_dir, destination_dir):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".md"):
            source_path = os.path.join(source_dir, filename)
            # Use nbconvert to convert markdown to notebook
            subprocess.run([
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",  # optional: remove this line if you don't want to execute the notebook
                source_path,
                "--output-dir", destination_dir
            ])
            print(f"Converted: {filename}")

# Example usage
source_directory = "./notes"
destination_directory = "./stub"
convert_md_to_ipynb(source_directory, destination_directory)
