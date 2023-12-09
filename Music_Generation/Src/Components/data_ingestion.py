import os
from Logger import logger 
import requests
from pathlib import Path
from github import ContentFile
from Entity.entity_config import (DataIngestionConfig)



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
      
        self.folder = config.folder
        self.repository = config.repository
        self.save_folder = config.save_folder

    
    def download(self,c: ContentFile, out: str):

        r = requests.get(c.download_url)
        output_path = Path(f'{out}/{c.path}')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            logger.info(f'downloading {c.path} to {out}')
            f.write(r.content)

    def git_download_folder(self):

        folder = self.repository.get_contents(self.folder.as_posix())

        for sub_folder in folder:

            if sub_folder.download_url is None:

                self.git_download_folder(self.repository,
                                        sub_folder.path,
                                        self.save_folder.as_posix())

                continue

            self.download(sub_folder,self.save_folder)        
        
    
