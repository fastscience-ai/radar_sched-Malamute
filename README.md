# radar_sched-Malamute
## Contributors ##
Paul Piccirillo, Stanford Research Institute

Vidyasagar Sadhu, Stanford Research Institute

Soo Kyung Kim, Palo Alto Research Center (= Stanford Resarch Institute)

## Installation ##
```
 conda create --name radar python=3.7
 conda activate radar
 conda install tensorflow==1.15.0
 pip install numpy simpy gym mpi4py stable_baselines
```

## Execution ##
### test environment ###
```
python env/main.py
```

### train ppo ###
```
python env/train_ppo.py 
```
