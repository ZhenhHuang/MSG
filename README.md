# [NeurIPS 2024] Spiking Graph Neural Networks on Riemannian Manifolds

## Get Started

Install Python packages.
```shell
pip install -r requirements.txt
```
Then you can run the command to train the model.
```shell
python main.py --task NC --dataset Physics --root_path ${your_path}
```

please replace ```${your_path}``` with your dataset file path.
If you want to use vallina SNN neurons, you can add ```--use_MS```. 

If you need to use product space of manifolds, you can add ```--use_product```.

All the configuration of models can be load from Json file in ```./configs```.

## Model Architecture

<div align=center>
<img src="./pics/model_1.png" width=80% alt="./pics/model_1.png" title="MSG" >
</div>

<div align=center>
<img src="./pics/model_2.png" width=80% alt="./pics/model_2.png" title="Manifold Spiking Layer" >
</div>

## Results
<div align=center>
<img src="./pics/results.png" width=100% alt="./pics/results.png" title="Sphere" >
</div>

## Visualization
<div align=center>
<img src="./pics/Torus.png" width=60% alt="./pics/Torus.png" title="Sphere" >
</div>
<div align=center>
Figure 1. Visualization of 34-th node on KarateClub dataset on Torus manifold.
</div>
<br><br>
<div align=center>
<img src="./pics/manifold_0.png" width=60% alt="./pics/manifold_0.png" title="Sphere" >
</div>
<div align=center>
Figure 2. Visualization of 1-th node on KarateClub dataset on Sphere manifold.
</div>
<br><br>
<div align=center>
<img src="./pics/manifold_16.png" width=60% alt="./pics/manifold_16.png" title="Sphere">
</div>
<div align=center>
Figure 3. Visualization of 17-th node on KarateClub dataset on Sphere manifold.
</div>