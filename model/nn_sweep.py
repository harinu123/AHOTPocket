import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AUROC
import argparse
import wandb
import random


def get_data(features, data_setup):
    prefix = "biolip_ttd"
    if data_setup == "augmented":
        prefix += "_augmented"
    elif data_setup == "ordered":
        prefix += "_ordered"
    elif data_setup == "nonzero":
        prefix += "_nonzero"
    featname = "_preds"
    if features == "esm_only":
        featname = "_embs"
    elif features == "combined":
        featname = "_combined"

    train_X = np.load("%s_train%s_X.npy" % (prefix, featname))
    train_Y = np.load("%s_train_Y.npy" % prefix)
    val_X = np.load("%s_val%s_X.npy" % (prefix, featname))
    val_Y = np.load("%s_val_Y.npy" % prefix)
    # test_X = np.load("%s_test%s_X.npy" % (prefix,featname))
    # test_Y = np.load("%s_test_Y.npy" % prefix)

    train_X = torch.tensor(train_X.reshape((train_X.shape[0], -1)), dtype=torch.float32)
    train_Y = torch.tensor(train_Y.reshape((train_X.shape[0], -1)), dtype=torch.float32)
    val_X = torch.tensor(val_X.reshape((val_X.shape[0], -1)), dtype=torch.float32)
    val_Y = torch.tensor(val_Y.reshape((val_X.shape[0], -1)), dtype=torch.float32)
    # test_X = torch.tensor(test_X.reshape((test_X.shape[0], -1)), dtype=torch.float32)
    # test_Y = torch.tensor(test_Y.reshape((test_X.shape[0], -1)), dtype=torch.float32)

    # return train_X.float(), train_Y.float(), val_X.float(), val_Y.float(), test_X.float(), test_Y.float()
    return train_X.float(), train_Y.float(), val_X.float(), val_Y.float()


def get_model(config_dict):
    if config_dict["features"] == "preds_only":
        in_size = 7 * 15  # 7 predictions, 15 amino acids
    elif config_dict["features"] == "esm_only":
        in_size = 2560 * 15  # length 2560 embeddings, 15 amino acids
    elif config_dict["features"] == "combined":
        in_size = (7 + 2560) * 15  # concatenation of the above
    else:
        "WARNING: invalid feature set specified! aborting..."
        return
    layers = []
    steps = int(in_size ** (1 / (config_dict["n_hidden"] + 1)))
    for l_num in range(config_dict["n_hidden"]):
        out_size = int(in_size / steps)
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.Dropout(p=config_dict["dropout"]))
        layers.append(nn.ReLU())
        in_size = out_size
    layers.append(nn.Linear(in_size, 1))
    layers.append(nn.Sigmoid())
    model = nn.Sequential(*layers)
    print(model)
    return model


def main(config_dict=None, project=None, outname=None):
    if not config_dict:
        run = wandb.init()
        config = run.config
        config_dict = {
            "features": config.features,
            "n_hidden": config.n_hidden,
            "lr": config.lr,
            "batchsize": config.batchsize,
            "dropout": config.dropout,
            "opt": config.opt,
            "epochs": config.epochs,
            "seed": config.seed,
            "data_setup": config.data_setup,
        }
    else:
        wandb.init(project=project)

    random.seed(config_dict["seed"])
    torch.manual_seed(config_dict["seed"])

    # train_X, train_Y, val_X, val_Y, test_X, test_Y = get_data(features = config_dict["features"], augmented=augmented)
    train_X, train_Y, val_X, val_Y = get_data(
        features=config_dict["features"], data_setup=config_dict["data_setup"]
    )

    model = get_model(config_dict)

    loss_fn = nn.BCELoss()

    opt_dict = {"Adam": optim.Adam, "AdamW": optim.AdamW}
    optimizer = opt_dict[config_dict["opt"]](model.parameters(), lr=config_dict["lr"])

    batch_size = config_dict["batchsize"]

    metric = AUROC(task="binary")
    best_epoch = -1
    best_val_auc = -1

    for epoch in range(config_dict["epochs"]):
        model.train()
        for i in range(0, len(train_X), batch_size):
            Xbatch = train_X[i : i + batch_size]
            y_pred = model(Xbatch)
            ybatch = train_Y[i : i + batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})

        model.eval()
        with torch.no_grad():
            y_val_pred = model(val_X)
            val_loss = loss_fn(y_val_pred, val_Y)
            print(
                f"Finished epoch {epoch}, training loss {loss}, validation loss {val_loss}",
                flush=True,
            )
            wandb.log({"val_loss": val_loss, "epoch": epoch})

            y_train_pred = model(train_X)
            train_accuracy = (
                (y_train_pred.round().flatten() == train_Y.flatten()).float().mean()
            )
            train_auc = metric(y_train_pred.flatten(), train_Y.flatten())
            print(
                f"Training accuracy: {train_accuracy}, Training AUC: {train_auc}",
                flush=True,
            )
            wandb.log({"train_acc": train_accuracy, "train_auc": train_auc})

            val_accuracy = (
                (y_val_pred.round().flatten() == val_Y.flatten()).float().mean()
            )
            val_auc = metric(y_val_pred.flatten(), val_Y.flatten())
            print(
                f"Validation accuracy: {val_accuracy}, Validation AUC: {val_auc}",
                flush=True,
            )
            wandb.log({"val_acc": val_accuracy, "val_auc": val_auc})
            if val_auc > best_val_auc:
                best_epoch = epoch
                best_val_auc = val_auc

                if type(outname) == str:
                    torch.save(model.state_dict(), outname)

            wandb.log({"best_val_auc": best_val_auc})

    y_train_pred = model(train_X)
    train_accuracy = (
        (y_train_pred.round().flatten() == train_Y.flatten()).float().mean()
    )
    print(f"Training accuracy: {train_accuracy}", flush=True)

    train_zeros = (train_Y.flatten() == 0).float().mean()
    print(f"Training accuracy if always guessing zero: {train_zeros}")

    y_val_pred = model(val_X)
    val_accuracy = (y_val_pred.round().flatten() == val_Y.flatten()).float().mean()
    print(f"Validation accuracy: {val_accuracy}")

    val_zeros = (val_Y.flatten() == 0).float().mean()
    print(f"Validation accuracy if always guessing zero: {val_zeros}")

    print(f"Best validation AUC was {best_val_auc}, achieved in Epoch {best_epoch}")


if __name__ == "__main__":
    config_dict = {
        "batchsize": 64,
        "dropout": 0.4,
        "epochs": 50,
        "features": "combined",
        "lr": 0.01,
        "n_hidden": 1,
        "opt": "Adam",
        "seed": 733,
        "data_setup": "augmented",
    }
    outname = "nn_augmented_combined_prime-sweep-241_733.pt"
    project = "hotpocket-nn-sweep"

    main(config_dict=config_dict, project=project, outname=outname)
    # main()
