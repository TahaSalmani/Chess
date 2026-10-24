import os
from dataclasses import replace
from pathlib import Path

import torch
from Chess import logger
from Chess.config.configuration import ConfigurationManager
from Chess.components.policy_value_torch_model import ValuePolicyNet
from Chess.components.self_play import generate_self_play_data, save_self_play_data
from Chess.components.train_rl import train_rl_step

STAGE_NAME = "Reinforcement Learning Full Pipeline Stage"


def _load_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def _arch_overrides(state, base_cfg):
    """Detect the architecture a checkpoint was trained with, so the RL net can be
    built to match it WITHOUT editing the supervised params.yaml. Returns only the
    fields that actually differ from base_cfg."""
    ov = {}
    if "input_proj.weight" in state:
        w = state["input_proj.weight"]
        ov["params_D_MODEL"] = int(w.shape[0])
        ov["params_IN_CHANNELS"] = int(w.shape[1])
    if "out.bias" in state:
        ov["params_NUM_MOVES"] = int(state["out.bias"].shape[0])
    if "blocks.0.ffn.0.weight" in state:
        ov["params_FF_DIM"] = int(state["blocks.0.ffn.0.weight"].shape[0])
    n_layers = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    if n_layers:
        ov["params_NUM_LAYERS"] = n_layers
    return {k: v for k, v in ov.items() if getattr(base_cfg, k, None) != v}


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

        rl_weights_path = os.path.join(str(rl_config.root_dir), "rl_trained_model.pth")
        best_weights_path = str(Path(training_config.trained_model_path).with_name("best_model.pth"))
        supervised_weights_path = best_weights_path if os.path.exists(best_weights_path) else str(training_config.trained_model_path)

        # Pick the checkpoint to bootstrap from, then adapt the net architecture to
        # whatever that checkpoint actually is (params.yaml may declare a different
        # depth/width than the weights on disk).
        init_kind, init_path = None, None
        if os.path.exists(rl_weights_path):
            init_kind, init_path = "rl", rl_weights_path
        elif os.path.exists(supervised_weights_path):
            init_kind, init_path = "supervised", supervised_weights_path

        init_state = None
        if init_path is not None:
            init_state = _load_state(init_path, device)
            ov = _arch_overrides(init_state, base_model_config)
            if ov:
                logger.info(
                    f"Adapting RL net to checkpoint architecture {ov} "
                    f"(params.yaml declares IN_CHANNELS={base_model_config.params_IN_CHANNELS}, "
                    f"NUM_LAYERS={base_model_config.params_NUM_LAYERS}); supervised config left unchanged."
                )
                base_model_config = replace(base_model_config, **ov)

        model = ValuePolicyNet(config=base_model_config).to(device)

        if init_kind == "rl":
            logger.info(f"Loading previous RL weights from: {init_path}")
            model.load_state_dict(init_state)
        elif init_kind == "supervised":
            logger.info(f"Loading supervised weights from: {init_path}")
            missing, unexpected = model.load_state_dict(init_state, strict=False)
            # value_fc/value_out are expected to be absent on the first RL bootstrap;
            # anything else missing/unexpected means the checkpoint does not match.
            unexpected_missing = [k for k in missing if not k.startswith(("value_fc", "value_out"))]
            if unexpected_missing or unexpected:
                raise RuntimeError(
                    f"{init_path} does not match ValuePolicyNet: missing={unexpected_missing[:3]}, "
                    f"unexpected={list(unexpected)[:3]}. Re-run stage 09 instead of bootstrapping RL from it."
                )
            value_keys = [k for k in missing if k.startswith(("value_fc", "value_out"))]
            policy_loaded = len([k for k in init_state if k in model.state_dict()])
            logger.info(
                f"Pretrained policy loaded OK: {policy_loaded} tensors from checkpoint; "
                f"value head randomly initialised ({len(value_keys)} tensors: {value_keys})."
            )
        else:
            logger.info("No pre-trained weights found. Starting with base model initial weights.")

        logger.info(
            f"MCTS config: {rl_config.params_num_simulation} simulations | "
            f"{rl_config.params_num_games} games | max {rl_config.params_max_moves} moves | "
            f"pool size {rl_config.params_pool_size}"
        )

        iterations = max(1, int(getattr(rl_config, "params_rl_iterations", 1)))
        logger.info(f"RL loop: {iterations} iteration(s) of self-play -> train")

        for it in range(1, iterations + 1):
            logger.info(f">>> RL iteration {it}/{iterations}: self-play data generation <<<")
            self_play_data = generate_self_play_data(model=model, device=device, config=rl_config)
            if not self_play_data:
                logger.warning(f"Iteration {it} produced no self-play data; stopping the loop.")
                break
            save_self_play_data(data=self_play_data, config=rl_config)

            logger.info(f">>> RL iteration {it}/{iterations}: training step <<<")
            # train_rl_step updates `model` in place, so the next iteration's MCTS
            # uses the improved policy AND value head (the value head bootstraps
            # from noise over rounds instead of staying random after one pass).
            train_rl_step(model=model, config=rl_config, device=device)

        logger.info(
            "RL training finished. rl_trained_model.pth is an UNADOPTED CANDIDATE. "
            "Run the real gate to decide adoption: "
            "`python research/rl_gate.py --games 20 --min-score 0.55`. The gate plays the "
            "candidate (MCTS) against the supervised policy+search champion and only promotes "
            "it (to promoted_model.pth) if it meets the acceptance score; on rejection the "
            "supervised best_model.pth is left byte-for-byte unchanged."
        )


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
        obj = TrainRLPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} Completed Successfully! <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e