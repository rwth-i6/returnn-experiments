For the paper
"Monotonic segmental attention for automatic speech recognition".

To reproduce our experiments, you will need to create a Sisyphus (https://github.com/rwth-i6/sisyphus) setup.

Inside the "recipe" folder, you will need to clone the following repositories:

  - https://github.com/rwth-i6/i6_core
  - https://github.com/rwth-i6/i6_experiments
  
Inside the "config" folder, you will need to create an "\_\_init\_\_.py" file with the following content:

```
from recipe.i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.main_pipeline import run_pipeline


def main():
  run_pipeline()
```

Then, from inside the setup root folder, you need to call "./sis m" to start the training, recognition and all other
required jobs which are needed to reproduce the results from the paper.