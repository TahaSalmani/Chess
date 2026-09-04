import  os
from pathlib import Path
import logging
from src.Chess import logger

project_name = "Chess"

list_of_files = [
".github/workflows/ci.yaml",
f"src/{project_name}/__init__.py",
f"src/{project_name}/components/__init__.py",
f"src/{project_name}/utils/__init__.py",
f"src/{project_name}/config/__init__.py",
f"src/{project_name}/config/configuration.py",
f"src/{project_name}/pipeline/__init__.py",
f"src/{project_name}/entity/__init__.py",
f"src/{project_name}/constants/__init__.py",
"dvc.yaml" ,
"params.yaml" ,
"requirements.txt" ,
"setup.py" ,
"research/notebooks.ipynb" ,
"templates/index.html" ,

]

for filepath in list_of_files :
    file_path = Path(filepath)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "" :
        os.makedirs(file_dir ,exist_ok=True)
        logging.info(f"Creating {file_path}")
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0) :
        with open(file_path,"w") as f:
            pass
            logging.info(f"creating empty file  {file_path}")
    else :
        logging.info(f"file Created   {file_path}")
