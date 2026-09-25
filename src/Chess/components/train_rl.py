from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from Chess.components.RL_Env import ChessEnv
from Chess.config.configuration import ConfigurationManager
from Chess.config.configuration import PrepareRlLearnConfig
from Chess import logger



class TrainRl :
    def __init__(self , config : PrepareRlLearnConfig) :
        self.config = config

        config_manager = ConfigurationManager()

        self.env = ChessEnv( config = config_manager.get_Rl_env() )

    def train  (self) :
        logger.info("Starting RL Training with MaskablePPO...")

        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env=self.env,
            verbose=self.config.verbose,
            learning_rate=self.config.learning_rate,

            tensorboard_log=str(
                self.config.tensorboard_log_dir
            ),

        )
        model.learn(self.config.total_timesteps , tb_log_name="ppo_chess_run")
        model.save(self.config.root_dir)
        logger.info("RL Training completed and model saved successfully!")