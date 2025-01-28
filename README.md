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
    $ cd <this repository>/src/unittest
    $ python -m unittest
    ```

* training
    ```bash
    # modify the configuration in scripts/configs/train.yml
    $ python <this repository>/src/scripts/train.py \
        <this repository>/src/scripts/configs/train.yml
    ```

* fine-tune
    ```bash
    # modify the configuration in scripts/configs/fine_tune.yml
    $ python <this repository>/src/scripts/fine_tune.py \
        <this repository>/src/scripts/configs/fine_tune.yml
    ```

* evaluation
    ```bash
    $ python <this repository>/src/scripts/evaluate.py \
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

* Train on VL-CMU-CD

| name             | train on VL-CMU-CD    | train on diff VL-CMU-CD   | fine-tune on PSCD   |
| ---------------- | :-------------------: | :-----------------------: | :-----------------: |
| ours (DinoV2)    | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.CMU.pth) | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.Diff-CMU.pth) | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.PSCD.pth) |
| ours (Resnet-18) | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/resnet18.2CrossAttn.CMU.pth) | / | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/resnet18.2CrossAttn.PSCD.pth) |
| C-3PO            | [resnet18_id_4_deeplabv3_VL_CMU_CD](https://github.com/DoctorKey/C-3PO) |                     |                     |
| DR-TANet         |                     |                     |                     |
| CDNet            |                     |                     |                     |
| TransCD          | [VL-CMU-CD -> Res-SViT_E1_D1_16.pth](https://github.com/wangle53/TransCD) |                     |                     |

* backbone v.s. comparator

| backbone  | comparator         | train on VL-CMU-CD |
| --------- | ------------------ | ------------------ |
| DinoV2    | Co-Attention       | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.CoAttn.CMU.pth) |
| DinoV2    | Temporal Attention | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.TemporalAttn.CMU.pth) |
| DinoV2    | MTF                | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.MTF.CMU.pth) |
| DinoV2    | 1 CrossAttn        | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.1CrossAttn.CMU.pth) |
| DinoV2    | 2 CrossAttn        | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.CMU.pth) |
| Resnet-18 | 2 CrossAttn        | [weight](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/resnet18.2CrossAttn.CMU.pth) |

### TODO

* [x] release source code
    * [x] release datasets module
    * [x] release models module
    * [x] release train/fine-tune/evaluation/visualize scripts
* [x] release pretraining weight
* [ ] examples of inference on new scenes
