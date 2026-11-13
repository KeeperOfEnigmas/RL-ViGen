from pathlib import Path

folder = Path(r"D:\Git\RL-ViGen\exp_local\pieg")

if not folder.is_dir():
    raise NotADirectoryError(f"Folder not found: {folder}")

for file in folder.rglob("*"):
    if file.is_file() and file.suffix.lower() == ".mp4":
        try:
            file.unlink()
            print(f"Deleted: {file}")
        except OSError as error:
            print(f"Could not delete {file}: {error}")