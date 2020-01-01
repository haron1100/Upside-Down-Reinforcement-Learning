# Upside-Down-Reinforcement-Learning
Implementation of Schmidhuber's Upside Down Reinforcement Learning paper

Link to paper with theory: https://arxiv.org/pdf/1912.02875.pdf

Link to paper with implementation details and results: https://arxiv.org/pdf/1912.02877.pdf

Use as you wish. Tweet(@mfharoon)/email(hshams@hotmail.co.uk) me any interesting results you find and sets of hyperparameters that work for particular environments. I will share here. Thanks!

### Working Parameters

#### CartPole
replay_size = 600<br>
last_few = 50<br>
batch_size = 64<br>
n_warm_up_episodes = 50<br>
n_episodes_per_iter = 50<br>
n_updates_per_iter = 100<br>
command_scale = 0.02<br>
lr = 0.001

