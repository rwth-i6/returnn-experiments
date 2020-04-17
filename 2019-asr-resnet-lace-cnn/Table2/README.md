Configuration files for the experiments from Table 3 in the paper:

#### Effect of the training method and the learning rate in ResNet and LACE.

| Model | Method | Mom. value | Epsilon | Learning rate | WER [%] |
| :----------- | ----------: | ----------: | ----------: | ----------: | ----------: |
| ResNet | Nadam | 0.9 | 1.0 | 10−3 | 17.5 |
| ResNet | Nadam | 0.9 | 1.0 | 0.5 · 10−3 | 17.0 |
| ResNet | Nadam | 0.9 | 1.0 | 10−4 | 18.7 |
| ResNet | SGD | 0.99 | - | 10−5 | 16.8 |
| ResNet | SGD | 0.99 | - | 0.5 · 10−5 | 16.6 |
| ResNet | SGD | 0.99 | - | 10−6 | 17.5 |
| LACE | Nadam | 0.9 | 1.0 | 10−3 | 16.1 |
| LACE | Nadam | 0.9 | 1.0 | 0.5 · 10−3 | 15.8 |
| LACE | Nadam | 0.9 | 1.0 | 10−4 | 16.5 |
| LACE | Nadam | 0.9 | 0.1 | 0.5 · 10−3 | 16.5 |
| LACE | SGD | 0.99 | - | 10−5 | 15.7 |
| LACE | SGD | 0.99 | - | 0.5 · 10−5 | 15.5 |
| LACE | SGD | 0.99 | - | 10−6 | 16.0 |
| LACE | SGD | 0.95 | - | 0.5 · 10−5 | 16.1 |
| LACE | SGD | 0.9 | - | 0.5 · 10−5 | 16.8 |
