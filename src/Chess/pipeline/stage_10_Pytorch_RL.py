import os
import torch
from Chess import logger
from Chess.config.configuration import ConfigurationManager
from Chess.components.ChessEnv import ChessEnv
from Chess.components.policy_value_torch_model import ValuePolicyNet
from Chess.components.Mcts import MCTS
from Chess.components.self_play import generate_self_play_data, save_self_play_data
from Chess.components.train_rl import train_rl_step

STAGE_NAME = "Reinforcement Learning Full Pipeline Stage"


class TrainRLPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def main(self):
        logger.info("Fetching configurations for RL pipeline...")
        base_model_config = self.config_manager.get_torch_base_model()
        training_config = self.config_manager.get_torch_training_config()
        rl_config = self.config_manager.get_rl_config()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        env = ChessEnv(config=None)
        model = ValuePolicyNet(config=base_model_config).to(device)

        weights_path = training_config.trained_model_path
        if os.path.exists(weights_path):
            logger.info(f"Loading existing weights from: {weights_path}")
            model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        else:
            logger.info("No pre-trained weights found. Starting with base model initial weights.")


        mcts = MCTS(model=model, device=device, config=rl_config)
        logger.info(f"MCTS initialized with {rl_config.params_num_simulation} simulations.")

        self_play_data_dir = os.path.join("artifacts", "self_play_data")
        logger.info(">>> Stage 1: Starting Self-Play Data Generation <<<")

        self_play_data = generate_self_play_data(
            self=self,
            model=model,
            device=device,

        )

        save_self_play_data(data=self_play_data, save_dir=self_play_data_dir)

        logger.info(">>> Stage 2: Starting RL Training Step <<<")
        train_rl_step(
            model=model,
            data_dir=self_play_data_dir,
            config=training_config,
            device=device
        )

        logger.info("RL Training completed successfully.")


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
        obj = TrainRLPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} Completed Successfully! <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e