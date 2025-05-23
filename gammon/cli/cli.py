import argparse
from pathlib import Path
from .post import post_process


def main():
    parser = argparse.ArgumentParser(description="CLI tools for gammon.")
    parser.add_argument("folder", type=str, help="Path to the Results folder")
    args = parser.parse_args()
    folder_path = Path(args.folder)

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Wrong Path to Results Folder: '{folder_path}'")
        return

    # Check for required files
    params_file = folder_path / "GCMC.params"
    out_file = folder_path / "GCMC.out"

    if not params_file.exists():
        print(f"Error: GCMC.params not in folder '{params_file.name}'.")
        return
    if not out_file.exists():
        print(f"Error: GCMC.out not in folder '{out_file.name}'.")
        return

    process_files(params_file, out_file)


def process_files(params_file, out_file):
    with open(params_file, "r") as params:
        paramslines = params.readlines()

    for line in paramslines:
        if line.startswith("Mu :"):
            mu = eval(line.split(":", 1)[1].strip())
            break

    post_process(out_file, mu)


if __name__ == "__main__":
    main()
