import os
import subprocess

def convert_ipynb_to_md(source_dir, destination_dir):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # loop through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".ipynb"):
            source_path = os.path.join(source_dir, filename)
            # Use nbconvert to convert to markdown
            subprocess.run([
                "jupyter", "nbconvert",
                "--to", "markdown",
                source_path,
                "--output-dir", destination_dir
            ])
            print(f"Converted: {filename}")

# Example usage
source_directory = "./notes"
destination_directory = "./stub"
convert_ipynb_to_md(source_directory, destination_directory)