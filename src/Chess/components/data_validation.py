import chess.pgn
from Chess import logger
import os
from pathlib import Path
from Chess.entity.config_entity import DataValidationConfig
import json
from datetime import datetime


class DataValidation:
    def __init__(self , config : DataValidationConfig):
        self.config = config


    def validate(self) -> bool:
        try :
            validation_status =  False
            os.makedirs(os.path.dirname(self.config.status_file), exist_ok=True)

            if not os.path.exists(self.config.source_file) or (os.path.getsize(self.config.source_file) == 0):
                validation_status = False
                logger.error(f"Source file not exist{self.config.source_file}")

            else :
                with open(self.config.source_file, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                    game = chess.pgn.read_headers(pgn_file)

                if game is not None and "Event" in game :
                    validation_status = True
                else:
                    validation_status = False
            status_data = {
                "validation_status" : validation_status,
                "Checked_Time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            }

            with open (self.config.status_file, "w", encoding="utf-8") as status_file :
                json.dump(status_data, status_file, indent=4)

            with open (self.config.status_file , 'w') as status_file:
                status_file.write(f"status is {validation_status}")

                logger.info(f"Status file saved at {self.config.status_file}")
                return validation_status

        except Exception as e:
            logger.exception(e)
            raise e