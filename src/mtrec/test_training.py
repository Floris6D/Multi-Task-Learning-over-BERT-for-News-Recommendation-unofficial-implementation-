from mtrec_class import Mtrec
from utils import load_configuration, get_dataloaders
from ebrec.utils._python import write_submission_file, rank_predictions_by_score
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
import polars as pl
import torch

cfg = load_configuration('test')
device = "cuda" if torch.cuda.is_available() else "cpu"
Mtrec_model = Mtrec(cfg, device="cpu").to(device)
(dataloader_train, dataloader_val) = get_dataloaders(cfg)


# Just for debugging this script
print("Check 1")
_ = Mtrec_model.train(dataloader_train)


