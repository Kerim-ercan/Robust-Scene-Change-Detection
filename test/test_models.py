import itertools
import unittest

import torch

import robust_scene_change_detect.models as models


class TestEncoder(unittest.TestCase):

    def test_dino_encoder(self):

        options = [
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ]

        for output in itertools.product([0, 2], [True, False], options):

            layer, freeze, model = output

            models.get_dino_backbone(
                **{
                    "dino-model": model,
                    "freeze-dino": freeze,
                    "unfreeze-dino-last-n-layer": layer,
                }
            )

    def test_resnet_encoder(self):

        options = [
            "resnet18",
            "resnet50",
        ]

        for option in options:
            models.ResNet(option)


class TestCDModels(unittest.TestCase):

    dino_options = [
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
    ]

    universal_dino_options = {
        "layer1": 11,
        "facet1": "query",
        "facet2": "query",
        "num-heads": 1,
        "dropout-rate": 0.1,
        "target-shp-row": 504,
        "target-shp-col": 504,
        "num-blocks": 1,
        "dino-model": "dinov2_vits14",
        "freeze-dino": True,
        "unfreeze-dino-last-n-layer": 0,
    }

    resnet_options = {
        "num-heads": 1,
        "dropout-rate": 0.1,
        "target-shp-row": 512,
        "target-shp-col": 512,
        "target-feature": 128,
    }

    def test_dino2_cross_attention(self):

        self.universal_dino_options["name"] = "dino2 + cross_attention"

        M = models.get_model(**self.universal_dino_options).to("cuda")

        with torch.no_grad():
            x0 = torch.randn(1, 3, 504, 504).to("cuda")
            x1 = torch.randn(1, 3, 504, 504).to("cuda")

            out = M(x0, x1)
            self.assertEqual(out.shape, (1, 504, 504, 2))

    def test_dino2_single_cross_attention1(self):

        self.universal_dino_options["name"] = "dino2 + single_cross_attention1"

        M = models.get_model(**self.universal_dino_options).to("cuda")

        with torch.no_grad():
            x0 = torch.randn(1, 3, 504, 504).to("cuda")
            x1 = torch.randn(1, 3, 504, 504).to("cuda")

            out = M(x0, x1)
            self.assertEqual(out.shape, (1, 504, 504, 2))

    def test_dino2_merge_temporal(self):

        self.universal_dino_options["name"] = "dino2 + merge_temporal"

        M = models.get_model(**self.universal_dino_options).to("cuda")

        with torch.no_grad():
            x0 = torch.randn(1, 3, 504, 504).to("cuda")
            x1 = torch.randn(1, 3, 504, 504).to("cuda")

            out = M(x0, x1)
            self.assertEqual(out.shape, (1, 504, 504, 2))

    def test_dino2_co_attention(self):

        self.universal_dino_options["name"] = "dino2 + co_attention"

        M = models.get_model(**self.universal_dino_options).to("cuda")

        with torch.no_grad():
            x0 = torch.randn(1, 3, 504, 504).to("cuda")
            x1 = torch.randn(1, 3, 504, 504).to("cuda")

            out = M(x0, x1)
            self.assertEqual(out.shape, (1, 504, 504, 2))

    def test_dino2_temporal_attention(self):

        self.universal_dino_options["name"] = "dino2 + temporal_attention"

        M = models.get_model(**self.universal_dino_options).to("cuda")

        with torch.no_grad():
            x0 = torch.randn(1, 3, 504, 504).to("cuda")
            x1 = torch.randn(1, 3, 504, 504).to("cuda")

            out = M(x0, x1)
            self.assertEqual(out.shape, (1, 504, 504, 2))

    def test_resnet18_cross_attention(self):

        self.resnet_options["name"] = "resnet18 + cross_attention"

        M = models.get_model(**self.resnet_options).to("cuda")

        with torch.no_grad():
            x0 = torch.randn(1, 3, 512, 512).to("cuda")
            x1 = torch.randn(1, 3, 512, 512).to("cuda")

            out = M(x0, x1)
            self.assertEqual(out.shape, (1, 512, 512, 2))
