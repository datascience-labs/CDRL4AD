import json
from tqdm import tqdm
from eval_methods import *
from utils import *
import pandas as pd
import numpy as np
import time

class Predictor:
    def __init__(self, device, model, window_size, n_features, batch_size, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = batch_size
        self.pred_args = pred_args
        self.topk = pred_args['topk']
        self.summary_file_name = summary_file_name
        self.device = device

    def get_score(self, values, true_anomalies=None):   
        
        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        prediction = []
        reconstruction = []
        pred_err = []
        recon_err = []
        count = 0
        last = 1
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            x = x.permute(0, 2, 1)
            y = y.permute(0, 2, 1)

            preds, recons = self.model(x, y) 
            forecast_loss = torch.sqrt((y.squeeze(-1) - preds.squeeze(-1)) ** 2)  
            recon_loss = torch.sqrt((x.squeeze(-1) - recons.squeeze(-1)) ** 2)     
            
            prediction.append(preds.detach().cpu().numpy())
            reconstruction.append(preds.detach().cpu().numpy())
            pred_err.append(forecast_loss.detach().cpu().numpy())
            recon_err.append(recon_loss.detach().cpu().numpy())

        pred_b_err = np.concatenate(pred_err, axis=0) 
        recon_b_err = np.concatenate(recon_err, axis=0)  
        prediction = pd.DataFrame(np.concatenate(prediction, axis=0))
        reconstruction = pd.DataFrame(np.concatenate(reconstruction, axis=0))
        recon_b_err = recon_b_err.mean(axis=-1)
        
        anomaly_scores = pred_b_err + self.gamma * recon_b_err

        if self.scale_scores:
            q75, q25 = np.percentile(anomaly_scores, [75, 25])
            iqr = q75 - q25
            median = np.median(anomaly_scores)
            anomaly_scores = (anomaly_scores - median) / (1+iqr)

        df_dict = {}
        for i in range(preds.shape[1]):
            df_dict[f"True_{i}"] = values[self.window_size:, i]
            df_dict[f"A_Score_{i}"] = anomaly_scores[:, i]

        df = pd.DataFrame(df_dict)

        return df, anomaly_scores, prediction, reconstruction

    def predict_anomalies(self, train, test, true_anomalies, root_cause_labels=None, load_scores=False, save_output=True,
                          scale_scores=False):

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(f"output/{self.dataset}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"output/{self.dataset}/test_output.pkl")

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

        else:
            train_pred_df, train_anomaly_scores, _, _ = self.get_score(train)  
            test_pred_df, test_anomaly_scores, test_preds, test_recons = self.get_score(test, true_anomalies)
            
            train_anomaly_scores = adjust_anomaly_scores(train_anomaly_scores, self.dataset, True, self.window_size)
            test_anomaly_scores = adjust_anomaly_scores(test_anomaly_scores, self.dataset, False, self.window_size)

            root_cause_score = test_anomaly_scores.copy() 
            test_anomaly_scores = np.mean(test_anomaly_scores, axis=1)

            train_anomaly_scores = normalize_anomaly_scores(train_anomaly_scores)
            test_anomaly_scores = normalize_anomaly_scores(test_anomaly_scores)


        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        out_dim = self.n_features if self.target_dims is None else len(self.target_dims)
        all_preds = np.zeros((len(test_pred_df), out_dim))


        true_anomalies = np.array(true_anomalies).reshape((-1))
        bf_eval = bf_search(test_anomaly_scores, true_anomalies, self.dataset, start=0, end=1, step_num=100, verbose=False) 
        for k, v in bf_eval.items():
            bf_eval[k] = v 
            
        test_pred_df["A_Score_Global"] = test_anomaly_scores
        test_pred_df["A_True_Global"] = true_anomalies
        test_pred_df["Thresh_Global"] = bf_eval['threshold']


        test_pred_df.to_csv(f"{self.save_path}/test_output.csv")
        test_preds.to_csv(f"{self.save_path}/test_preds.csv")
        test_recons.to_csv(f"{self.save_path}/test_recons.csv")

        np.savetxt(f"{self.save_path}/root_cause_score.csv", root_cause_score, delimiter=",", fmt="%.6f")
        
        if self.dataset == 'SMD':
            hitrate = hit_att(root_cause_score, root_cause_labels)
            ndcg_score = ndcg(root_cause_score, root_cause_labels)
            print(hitrate)
            print(ndcg_score)
        print(bf_eval)

        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            f.write(f"Accuracy: {str(bf_eval)}\n")
            if self.dataset == 'SMD':
                f.write(f"Hitrate: {str(hitrate)}\n")
                f.write(f"NDCG Score: {str(ndcg_score)}\n")

        print(f"Results saved to {self.save_path}")
        print("-- Done.")
                            
