from mtrec_class import Mtrec
from utils import load_configuration, get_test_dataloader
from ebrec.utils._python import write_submission_file, rank_predictions_by_score
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
import polars as pl
import torch 
import argparse

# Initilize config file
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--file', default='test', help='Path to the configuration file')
#parser.add_argument('--run_name', default='Standard_model', help='Name of the run')
args = parser.parse_args()
# Load configuration, model and dataloaders
cfg = load_configuration(args.file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Mtrec_model = Mtrec(cfg, device=device).to(device)
dataloader_test = get_test_dataloader(cfg)

# Get predictions
Mtrec_model.load_checkpoint("saved_models/" + cfg['name_run'])
predictions = Mtrec_model.predict(dataloader_test)

# Evaluate predictions
metrics = MetricEvaluator(
    labels=predictions["labels"].to_list(),
    predictions=predictions["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
print(metrics.evaluate())

# Also store the metrics in a file
with open("saved_models/" + cfg['name_run'] + f"/Metrics_{cfg['name_run']}.txt", "w") as file:
    file.write(metrics.evaluate())

# Write submission file
predictions = predictions.with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)

write_submission_file(
    impression_ids=predictions['impression_id'],
    prediction_scores=predictions["ranked_scores"],
    path="saved_models/" + cfg['name_run'] + f"/predictions_{cfg['name_run']}.txt",
)
