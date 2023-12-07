import os
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s')


PROJECT_NAME = 'Music_Generation'

LIST_OF_FILES= [

    f".github/workflows/.gitkeep",
    f"{PROJECT_NAME}/Config.yaml",
    f"{PROJECT_NAME}/Params.yaml",
    f"{PROJECT_NAME}/Src/__init__.py",
    f"{PROJECT_NAME}/Src/Constants.py",
    f"{PROJECT_NAME}/Src/Utils.py",
    f"{PROJECT_NAME}/Src/Logger.py",
    f"{PROJECT_NAME}/Src/Exception.py",
    f"{PROJECT_NAME}/Src/Entity/__init__.py",
    f"{PROJECT_NAME}/Src/Entity/entity_config.py",
    f"{PROJECT_NAME}/Src/Configuration/__init__.py",
    f"{PROJECT_NAME}/Src/Configuration/Config.py",
    f"{PROJECT_NAME}/Src/Components/__init__.py",
    f"{PROJECT_NAME}/Src/Pipelines/__init__.py",
    f"{PROJECT_NAME}/Src/main.py",
    f"{PROJECT_NAME}/Src/app.py",
    f"{PROJECT_NAME}/dvc.yaml",
    f"{PROJECT_NAME}/Src/Setup.py",
    f"{PROJECT_NAME}/requirements.txt",
    f"{PROJECT_NAME}/Research/Trials.ipynb"

]

for file in LIST_OF_FILES:

    file_path = Path(file)
    file_directory,file_name = os.path.split(file_path)
 
    if file_directory!="":
        os.makedirs(file_directory,exist_ok=True)
        logging.info(f"Created directory: {file_directory} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            logging.info(f"Creating empty file : {file_path}")
    else:
        logging.info(f"{file_name} already exists")