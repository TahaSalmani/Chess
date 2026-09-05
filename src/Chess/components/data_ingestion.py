import os
import urllib.request as request
import zstandard as zstd
from pathlib import Path

from Chess import logger
from Chess.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info("DataSet is Downloading...")
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info("Download Completed.")
        else:
            logger.info("DataSet is already downloaded.")

    def extract_zst_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        output_pgn_file = os.path.join(unzip_path, "raw_games.pgn")

        logger.info("Extracting ZST file...")

        dctx = zstd.ZstdDecompressor()

        with open(self.config.local_data_file, "rb") as compressed_file:
            with open(output_pgn_file, "wb") as decompressed_file:
                dctx.copy_stream(compressed_file, decompressed_file)

        logger.info(f"Extracted ZST file successfully to {output_pgn_file}")