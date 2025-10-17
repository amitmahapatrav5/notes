import os
import subprocess
import nbformat
import re

def convert_ipynb_to_md(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.endswith(".ipynb"):
            source_path = os.path.join(source_dir, filename)
            subprocess.run([
                "jupyter", "nbconvert",
                "--to", "markdown",
                source_path,
                "--output-dir", destination_dir
            ])
            print(f"✅ Converted notebook to markdown: {filename}")


def convert_md_to_ipynb(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.endswith(".md"):
            source_path = os.path.join(source_dir, filename)
            dest_filename = os.path.splitext(filename)[0] + ".ipynb"
            dest_path = os.path.join(destination_dir, dest_filename)

            with open(source_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            nb = nbformat.v4.new_notebook()
            cell_lines = []
            in_code_block = False
            code_language = ""

            for line in lines:
                code_start = re.match(r"```(\w*)", line.strip())
                code_end = line.strip() == "```"

                if code_start and not in_code_block:
                    # Starting a code block
                    in_code_block = True
                    code_language = code_start.group(1)
                    if cell_lines:
                        # Add pending markdown cell before code block
                        nb.cells.append(nbformat.v4.new_markdown_cell("".join(cell_lines).strip()))
                        cell_lines = []
                elif code_end and in_code_block:
                    # End of code block
                    in_code_block = False
                    nb.cells.append(nbformat.v4.new_code_cell("".join(cell_lines).strip()))
                    cell_lines = []
                else:
                    cell_lines.append(line)

            # Add any remaining markdown at the end
            if cell_lines:
                nb.cells.append(nbformat.v4.new_markdown_cell("".join(cell_lines).strip()))

            with open(dest_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

            print(f"✅ Converted markdown to notebook: {filename}")


# Example usage
source_directory = "./notes"
destination_directory = "./stub"

convert_ipynb_to_md(source_directory, destination_directory)
convert_md_to_ipynb(source_directory, destination_directory)
