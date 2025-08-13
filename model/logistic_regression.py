import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import argparse
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


def assess_model(
    train_X,
    train_Y,
    val_X,
    val_Y,
    penalty,
    solver,
    seeds,
    Cs,
    l1s=[None],
    max_iter=100,
    verbose=False,
):
    train_acc_mean_list = []
    train_acc_std_list = []
    train_auc_mean_list = []
    train_auc_std_list = []
    val_acc_mean_list = []
    val_acc_std_list = []
    val_auc_mean_list = []
    val_auc_std_list = []
    c_list = []
    ratio_list = []
    for C in tqdm(Cs):
        for l1_ratio in l1s:
            train_accs = []
            train_aucs = []
            val_accs = []
            val_aucs = []
            for seed in seeds:
                model = LogisticRegression(
                    random_state=seed,
                    penalty=penalty,
                    max_iter=max_iter,
                    solver=solver,
                    l1_ratio=l1_ratio,
                    C=C,
                )
                model.fit(train_X, train_Y)
                train_X_pred = model.predict(train_X)
                val_X_pred = model.predict(val_X)

                train_accuracy = (train_X_pred.round() == train_Y).mean()
                val_accuracy = (val_X_pred.round() == val_Y).mean()
                train_auc = roc_auc_score(train_Y, train_X_pred)
                val_auc = roc_auc_score(val_Y, val_X_pred)

                train_accs.append(train_accuracy)
                train_aucs.append(train_auc)
                val_accs.append(val_accuracy)
                val_aucs.append(val_auc)

            train_accs = np.array(train_accs)
            train_aucs = np.array(train_aucs)
            val_accs = np.array(val_accs)
            val_aucs = np.array(val_aucs)

            train_acc_mean = np.mean(train_accuracy)
            train_acc_std = np.std(train_accuracy)
            train_auc_mean = np.mean(train_auc)
            train_auc_std = np.std(train_auc)
            val_acc_mean = np.mean(val_accuracy)
            val_acc_std = np.std(val_accuracy)
            val_auc_mean = np.mean(val_auc)
            val_auc_std = np.std(val_auc)

            train_acc_mean_list.append(train_acc_mean)
            train_acc_std_list.append(train_acc_std)
            train_auc_mean_list.append(train_auc_mean)
            train_auc_std_list.append(train_auc_std)
            val_acc_mean_list.append(val_acc_mean)
            val_acc_std_list.append(val_acc_std)
            val_auc_mean_list.append(val_auc_mean)
            val_auc_std_list.append(val_auc_std)

            c_list.append(C)
            ratio_list.append(l1_ratio)

            if verbose:
                print(
                    "Train accuracy: %f (std %f), Train AUC: %f (std %f)"
                    % (train_acc_mean, train_ac_std, train_auc_mean, train_auc_std)
                )
                print(
                    "Validation accuracy: %f (std %f), Validation AUC:  %f (std %f)"
                    % (val_acc_mean, val_acc_std, val_auc_mean, val_auc_std)
                )
    return pd.DataFrame(
        {
            "C": c_list,
            "l1 ratio": ratio_list,
            "Train acc mean": train_acc_mean_list,
            "Train acc std": train_acc_std_list,
            "Train auc mean": train_auc_mean_list,
            "Train auc std": train_auc_std_list,
            "Val acc mean": val_acc_mean_list,
            "Val acc std": val_acc_std_list,
            "Val auc mean": val_auc_mean_list,
            "Val auc std": val_auc_std_list,
        }
    )


def main(args):
    seeds = random.sample(range(1000), 2)

    datadir = "../data/model_inputs"

    if args.features == "pred":
        train_X = np.load(os.path.join(datadir, "biolip_ttd_train_preds_X.npy"))
        val_X = np.load(os.path.join(datadir, "biolip_ttd_val_preds_X.npy"))
        test_X = np.load(os.path.join(datadir, "biolip_ttd_test_preds_X.npy"))
    elif args.features == "embs":
        train_X = np.load(os.path.join(datadir, "biolip_ttd_train_embs_X.npy"))
        val_X = np.load(os.path.join(datadir, "biolip_ttd_val_embs_X.npy"))
        test_X = np.load(os.path.join(datadir, "biolip_ttd_test_embs_X.npy"))
    elif args.features == "comb":
        train_X = np.load(os.path.join(datadir, "biolip_ttd_train_combined_X.npy"))
        val_X = np.load(os.path.join(datadir, "biolip_ttd_val_combined_X.npy"))
        test_X = np.load(os.path.join(datadir, "biolip_ttd_test_combined_X.npy"))
    else:
        print(
            'ERROR. Feature option not supported. Please use "pred", "embs", or "comb"'
        )
        return

    train_Y = np.load(os.path.join(datadir, "biolip_ttd_train_Y.npy"))
    val_Y = np.load(os.path.join(datadir, "biolip_ttd_val_Y.npy"))
    test_Y = np.load(os.path.join(datadir, "biolip_ttd_test_Y.npy"))

    train_X = train_X.reshape((train_X.shape[0], -1))
    val_X = val_X.reshape((val_X.shape[0], -1))
    test_X = test_X.reshape((test_X.shape[0], -1))

    if args.features == "embs":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(train_X)

        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)
    elif args.features == "comb":
        train_X = train_X.reshape((train_X.shape[0], -1))
        val_X = val_X.reshape((val_X.shape[0], -1))
        test_X = test_X.reshape((test_X.shape[0], -1))
        train_X_to_scale = train_X[:, 7:]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(train_X_to_scale)

        train_X[:, 7:] = scaler.transform(train_X[:, 7:])
        val_X[:, 7:] = scaler.transform(val_X[:, 7:])
        test_X[:, 7:] = scaler.transform(test_X[:, 7:])

    Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    if args.i == 0:
        df = assess_model(
            train_X,
            train_Y,
            val_X,
            val_Y,
            penalty=None,
            solver="lbfgs",
            seeds=seeds,
            Cs=[0.1, 1],
        )
    elif args.i == 1:
        df = assess_model(
            train_X,
            train_Y,
            val_X,
            val_Y,
            penalty="l1",
            solver="liblinear",
            seeds=seeds,
            Cs=Cs,
        )
    elif args.i == 2:
        df = assess_model(
            train_X,
            train_Y,
            val_X,
            val_Y,
            penalty="l2",
            solver="lbfgs",
            seeds=seeds,
            Cs=Cs,
        )
    elif args.i == 3:
        df = assess_model(
            train_X,
            train_Y,
            val_X,
            val_Y,
            penalty="elasticnet",
            solver="saga",
            seeds=seeds,
            Cs=Cs,
            l1s=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
            max_iter=500,
        )
    else:
        print("detected i value of %d not accepted. aborting." % i)

    df.to_csv(args.outname, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", type=int, help="choice for logistic regression parameters"
    )
    parser.add_argument(
        "--features",
        type=str,
        help='what features to use; must be "pred", "embs", or "comb"',
    )
    parser.add_argument("--outname", type=str, help="name to save output csv")
    args = parser.parse_args()
    main(args)
