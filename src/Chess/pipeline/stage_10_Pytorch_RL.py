import os
from pathlib import Path

import torch
from Chess import logger
from Chess.config.configuration import ConfigurationManager
from Chess.components.policy_value_torch_model import ValuePolicyNet
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

        model = ValuePolicyNet(config=base_model_config).to(device)

        rl_weights_path = os.path.join(str(rl_config.root_dir), "rl_trained_model.pth")
        best_weights_path = str(Path(training_config.trained_model_path).with_name("best_model.pth"))
        supervised_weights_path = best_weights_path if os.path.exists(best_weights_path) else str(training_config.trained_model_path)
        if os.path.exists(rl_weights_path):
            logger.info(f"Loading previous RL weights from: {rl_weights_path}")
            model.load_state_dict(torch.load(rl_weights_path, map_location=device))
        elif os.path.exists(supervised_weights_path):
            logger.info(f"Loading supervised weights from: {supervised_weights_path}")
            missing, _ = model.load_state_dict(
                torch.load(supervised_weights_path, map_location=device), strict=False
            )
            # value_fc/value_out are expected to be absent on the first RL bootstrap;
            # anything else missing means the checkpoint does not match this architecture.
            unexpected_missing = [k for k in missing if not k.startswith(("value_fc", "value_out"))]
            if unexpected_missing:
                raise RuntimeError(
                    f"{supervised_weights_path} is missing {len(unexpected_missing)} keys that "
                    f"ValuePolicyNet expects (e.g. {unexpected_missing[0]}). It was saved by a "
                    f"different architecture or NUM_LAYERS={base_model_config.params_NUM_LAYERS} "
                    f"does not match. Re-run stage 09 instead of bootstrapping RL from it."
                )
        else:
            logger.info("No pre-trained weights found. Starting with base model initial weights.")

        logger.info(
            f"MCTS config: {rl_config.params_num_simulation} simulations | "
            f"{rl_config.params_num_games} games | max {rl_config.params_max_moves} moves | "
            f"pool size {rl_config.params_pool_size}"
        )

        logger.info(">>> Stage 1: Starting Self-Play Data Generation <<<")
        self_play_data = generate_self_play_data(model=model, device=device, config=rl_config)
        save_self_play_data(data=self_play_data, config=rl_config)

        logger.info(">>> Stage 2: Starting RL Training Step <<<")
        train_rl_step(model=model, config=rl_config, device=device)

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