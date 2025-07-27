# Instruction
Weilan Baby Alpha velocity command (linear and angular) following on flat terrain using rsl_rl.
## Training
To train, run the following command
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Baby-Alpha-Direct-v0 --headless
```
## Visualization
Visualize RL training using Tensorboard
```bash
tensorboard --logdir logs/rsl_rl/baby_alpha_flat_direct
```

To visualization the results:
```bash
 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Baby-Alpha-Direct-v0 --num_envs 10
```
