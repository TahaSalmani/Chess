from Chess.config.configuration import ConfigurationManager
from Chess import logger
from Chess.components.torch_Dataset import ChessDataset
STAGE_NAME = "TORCH DATASET "

class TorchDatasetPipelineStage:
    def __init__(self):
        pass
    def main(self):

        config_manager = ConfigurationManager()
        torch_dataset_config = config_manager.get_torch_dataset()
        torch_data = ChessDataset(config = torch_dataset_config )
        torch_data.save_data()


if __name__ == "__main__":
    try:
        logger.info("Torch Dataset pipeline stage started")
        stage = TorchDatasetPipelineStage()
        stage.main()
        logger.info("Torch Dataset pipeline stage finished")
    except Exception as e:
        logger.error(e)