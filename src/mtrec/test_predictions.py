from mtrec_class import Mtrec
from utils import load_configuration, get_dataloaders
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore

cfg = load_configuration('test')

Mtrec_model = Mtrec(cfg, device="cpu")
(dataloader_train, dataloader_val) = get_dataloaders(cfg)


print("Check 1")
Mtrec_model.load_checkpoint("saved_models/run1")
scores, labels = Mtrec_model.predict(dataloader_train)

print(f"Total datapoints: {len(dataloader_train.dataset)}")
print(f" Length of scores: {len(scores)}")
print(f"The first score: {scores[0]}")
print(f"The first label: {labels[0]}")

metrics = MetricEvaluator(
    labels=labels,
    predictions=scores,
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
print(metrics.evaluate())
