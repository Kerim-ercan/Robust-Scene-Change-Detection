import unittest

import torch

import robust_scene_change_detect.torch_utils as torch_utils


class TestTorchUtils(unittest.TestCase):

    def test_translate_image_1(self):

        img = torch.Tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )

        f = torch_utils.translate_image(1, 2)

        x = f(img)

        expected = torch.Tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 2],
            ]
        )

        self.assertTrue(torch.all(x == expected))

    def test_translate_image_2(self):

        img = torch.Tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )

        f = torch_utils.translate_image(-2, 1)

        x = f(img)

        expected = torch.Tensor(
            [
                [0, 0, 0],
                [3, 0, 0],
                [6, 0, 0],
            ]
        )

        self.assertTrue(torch.all(x == expected))


if __name__ == "__main__":
    unittest.main()
