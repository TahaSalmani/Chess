from Chess import  logger
from Chess.components.train_rl import TrainRl
from Chess.config.configuration import ConfigurationManager
STAGE_NAME = "RL"

class RlPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        rl_learn_config = config_manager.get_prepare_rl_learn()

        trainer = TrainRl(config=rl_learn_config)
        trainer.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = RlPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e