from mtrec_class import Mtrec
from utils import load_configuration, get_dataloaders
from ebrec.utils._python import write_submission_file, rank_predictions_by_score
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
import polars as pl
import torch

# Load configuration, model and dataloaders
cfg = load_configuration('test')
device = "cuda" if torch.cuda.is_available() else "cpu"
Mtrec_model = Mtrec(cfg, device="cpu").to(device)
(dataloader_train, dataloader_val) = get_dataloaders(cfg)

# Get predictions
print("Check 1")
Mtrec_model.load_checkpoint("saved_models/run1")
predictions = Mtrec_model.predict(dataloader_val)

# Evaluate predictions
metrics = MetricEvaluator(
    labels=predictions["labels"].to_list(),
    predictions=predictions["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
print(metrics.evaluate())

# Write submission file
predictions = predictions.with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)

write_submission_file(
    impression_ids=predictions['impression_id'],
    prediction_scores=predictions["ranked_scores"],
    path="saved_models/run1/predictions.txt",
)
