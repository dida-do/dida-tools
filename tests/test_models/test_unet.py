import unittest
import torch

from models.unet import UNET
from utils.torchutils import forward, backprop, assert_not_nan, assert_isfinite, assert_nonzero
from utils.loss import smooth_dice_loss

config = {
    "DEFAULT_WEIGHTS": None,
    "INPUT_DIM": (1, 12, 1024, 1024),
    "OUTPUT_DIM": (1, 2, 1024, 1024),
    "LOSS": smooth_dice_loss,
    "MODEL": UNET,
    "MODEL_CONFIG": {
        "ch_in" : 12,
        "ch_out" : 2,
        "n_recursions" : 5,
        "use_shuffle": True,
        "use_pooling": True
    },
    "OPTIMIZER": torch.optim.Adam,
    "OPTIMIZER_CONFIG": {
        "lr" : 10e-3
    },
    "DEVICE" : "cpu"
}

class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.device = config["DEVICE"]

        if config["DEFAULT_WEIGHTS"] is None:
            self._model = config["MODEL"](**config["MODEL_CONFIG"])
        else:
            self._model = load_model(config["MODEL"], config["MODEL_CONFIG"],
                                     config["DEFAULT_WEIGHTS"], config["DEVICE"])

    def test_parameters(self):
        for tensor in self._model.parameters():
            assert_not_nan(tensor)
            assert_isfinite(tensor)

    def test_forward(self):
        test_array = torch.randn(*config["INPUT_DIM"])
        pred = forward(self._model, test_array, self.device)
        assert tuple(pred.shape) == config["OUTPUT_DIM"]
        assert_isfinite(pred)
        assert_not_nan(pred)

class TestTraining(unittest.TestCase):

    def setUp(self):
        self.device = config["DEVICE"]

        if config["DEFAULT_WEIGHTS"] is None:
            self._model = config["MODEL"](**config["MODEL_CONFIG"])
        else:
            self._model = load_model(config["MODEL"], config["MODEL_CONFIG"],
                                     config["DEFAULT_WEIGHTS"], config["DEVICE"])

    def test_backprop_step(self):
        inputs = torch.randn(*config["INPUT_DIM"])
        targets = torch.randn(*config["OUTPUT_DIM"])
        optimizer = config["OPTIMIZER"](self._model.parameters(), **config["OPTIMIZER_CONFIG"])
        backprop(self._model, config["LOSS"], optimizer, [inputs, targets], self.device)
        for parameter in self._model.parameters():
            if parameter.requires_grad:
                assert_nonzero(parameter.grad)

if __name__ == "__main__":
    unittest.main()
