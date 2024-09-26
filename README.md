# Robust Scene Change Detection Using Visual Foundation Models and Cross-Attention Mechanisms


### Paper

* [arXiv link](http://arxiv.org/abs/2409.16850)

### An Introduction Video (3 minutes)

[![](https://img.youtube.com/vi/KX2E8Q5D-Fk/0.jpg)](https://www.youtube.com/watch?v=KX2E8Q5D-Fk)


### Installation

```bash
# create a Python 3.9.6 virtual environment
$ source <directory of virtual environment>/bin/activate
$ cd <this repository>
$ pip install -r requirements.txt
```

### Datasets

* download [VL-CMU-CD](https://huggingface.co/datasets/Flourish/VL-CMU-CD/blob/main/VL-CMU-CD-binary255.zip) & [PSCD](https://kensakurada.github.io/pscd/term_of_use.html) datasets

* update the both dataset directories to `datasets/data_factory`

### Example usage

* unittest
    ``` bash
    $ cd <this repository>/unittest
    $ python -m unittest
    ```

* training
    ```bash
    # modify the configuration in scripts/configs/train.yml
    $ python <this repository>/scripts/train.py \
        <this repository>/scripts/configs/train.yml
    ```

* fine-tune
    ```bash
    # modify the configuration in scripts/configs/fine_tune.yml
    $ python <this repository>/scripts/fine_tune.py \
        <this repository>/scripts/configs/fine_tune.yml
    ```

* evaluation
    ```bash
    $ python <this repository>/scripts/evaluate.py \
        <checkpoint directory>/<name>.pth
    ```

* qualitive results
    ```bash
    $ python <this repository>/scripts/visualize.py \
        <checkpoint directory>/best.val.pth \
        --option <option> \
        --output <directory for qualitive results>
    ```

    | options           | comments                            |
    | ----------------- | ----------------------------------- |
    | VL-CMU-CD         | aligned                             |
    | PSCD              | aligned                             |
    | VL-CMU-CD-diff_1  | unaligned (adjacent distance == 1)  |
    | VL-CMU-CD-diff_-1 | unaligned (adjacent distance == -1) |
    | VL-CMU-CD-diff_2  | unaligned (adjacent distance == 2)  |
    | VL-CMU-CD-diff_-2 | unaligned (adjacent distance == -2) |

### Pretrained Weight

* TBD

### TODO

* [ ] release source code
* [ ] release pretraining weight
* [ ] examples of inference on new scenes
