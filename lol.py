import shutil
from pathlib import Path

baseMaskPath = Path('largeDataset/NonProcessed/Masked')
newMaskPath = Path('largeDataset/Processed/Masked')


def copyfiles(path, newpath):
    folders = path.iterdir()
    for folder in folders:
        if folder.is_dir():
            folderPath = Path(path, folder.name)
            print(folderPath.name)
            files = folderPath.iterdir()
            for file in files:
                print(file.name)
                shutil.copy(Path(folderPath, file.name), newpath)



copyfiles(baseMaskPath, newMaskPath)
