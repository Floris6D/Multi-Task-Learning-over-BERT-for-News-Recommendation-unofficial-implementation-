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
#initialize optimizer
user_encoder = Mtrec_model.user_encoder
news_encoder = Mtrec_model.news_encoder
params = [
    {"params": [user_encoder.W,  user_encoder.q],   "lr": cfg["lr_user"]},  # lr of attention layer in user encoder
    {"params": list(news_encoder.cat_net.parameters()) + list(news_encoder.ner_net.parameters()),
                "lr": cfg["lr_news"]},  # lr of auxiliary tasks
    {"params": news_encoder.bert.parameters(), "lr": cfg["lr_bert"]}  # Parameters of BERT
] 

if cfg["optimizer"] == "adam":
    optimizer = torch.optim.Adam(params)
elif cfg["optimizer"] == "sgd":
    optimizer = torch.optim.SGD(params)
elif cfg["optimizer"] == "adamw":
    optimizer = torch.optim.AdamW(params)
else:
    print("Invalid optimizer <{}>.".format(cfg["optimizer"]))

#TURNEMALLON TODO: remove this?
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.requires_grad=True

print_flag = True
total_loss, total_main_loss = Mtrec_model.train(dataloader_train, optimizer, print_flag)

test = 7

