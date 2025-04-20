import gymnasium as gym

from . import agents

from .h1_soccer_env import H1SoccerEnvCfg

gym.register(
    id="Isaac-H1-Soccer-Direct-v0",
    entry_point=f"{__name__}.h1_soccer_env:H1SoccerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1SoccerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1PPORunnerCfg",
    },
)