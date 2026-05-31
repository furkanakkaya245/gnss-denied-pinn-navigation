import os
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


SEQ_LENGTH = 20
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


class AdvancedFusionNet(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
            batch_first=True
        )

        self.attention = AttentionLayer(128)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        context = self.attention(out)
        return self.fc(context)



def create_sequences(X, Y, seq_length=20):
    xs = []
    ys = []

    for i in range(len(X) - seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(Y[i+seq_length])

    return np.array(xs), np.array(ys)



def prepare_dataset(df, bias_x, bias_y):

    t = df['Zaman_sn'].values

    dt = np.diff(t, prepend=t[0])
    dt[0] = np.mean(dt[1:5]) if len(dt) > 5 else 0.02
    dt = np.clip(dt, 1e-4, None)

    # Bias correction
    df['acc_x_clean'] = df['acc_x'] - (bias_x[1] * t + bias_x[0])
    df['acc_y_clean'] = df['acc_y'] - (bias_y[1] * t + bias_y[0])

    # Integrated velocity
    df['vel_x_imu'] = np.cumsum(df['acc_x_clean'] * dt)
    df['vel_y_imu'] = np.cumsum(df['acc_y_clean'] * dt)

    # GNSS availability
    df['gnss_available'] = 1.0

    features = [
        'acc_x_clean',
        'acc_y_clean',
        'gyro_z',
        'vel_x_imu',
        'vel_y_imu',
        'gnss_x',
        'gnss_y',
        'gnss_Q',
        'gnss_available'
    ]

    feature_matrix = df[features].values

    # dt feature
    feature_matrix = np.column_stack([feature_matrix, dt])

    targets = np.stack([
        df['pos_x_true'].values,
        df['pos_y_true'].values
    ], axis=1)

    return feature_matrix, targets



def physics_loss(pred_pos, true_pos, seq_input):

    trajectory_loss = nn.MSELoss()(pred_pos, true_pos)

    # Approximate velocity continuity
    acc_x = seq_input[:, -1, 0]
    acc_y = seq_input[:, -1, 1]

    vel_constraint = torch.mean(acc_x**2 + acc_y**2)

    return trajectory_loss + 0.001 * vel_constraint



def train_advanced_model(base_dir):

    TRAIN_DATASETS = [1, 2, 3]
    VAL_DATASET = 4

    bias_x = (-0.10833055, 0.00094616)
    bias_y = (-0.10677880, -0.00074795)

    train_X = []
    train_Y = []

    scaler = StandardScaler()


    for i in TRAIN_DATASETS:

        file_path = os.path.join(
            base_dir,
            f"dataset{i}",
            f"dataset{i}_egitim_verisi.csv"
        )

        df = pd.read_csv(file_path)

        X, Y = prepare_dataset(df, bias_x, bias_y)

        train_X.append(X)
        train_Y.append(Y)

    X_train_full = np.concatenate(train_X)
    Y_train_full = np.concatenate(train_Y)

    scaler.fit(X_train_full)

    X_train_full = scaler.transform(X_train_full)

    joblib.dump(scaler, os.path.join(base_dir, "fusion_scaler.pkl"))

    X_train_seq, Y_train_seq = create_sequences(
        X_train_full,
        Y_train_full,
        SEQ_LENGTH
    )

    ########################################
    # VALIDATION DATA
    ########################################

    val_path = os.path.join(
        base_dir,
        f"dataset{VAL_DATASET}",
        f"dataset{VAL_DATASET}_egitim_verisi.csv"
    )

    val_df = pd.read_csv(val_path)

    X_val, Y_val = prepare_dataset(val_df, bias_x, bias_y)

    X_val = scaler.transform(X_val)

    X_val_seq, Y_val_seq = create_sequences(
        X_val,
        Y_val,
        SEQ_LENGTH
    )

    ########################################
    # TORCH
    ########################################

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_seq, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val_seq, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, Y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = AdvancedFusionNet(input_dim=10).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val = 1e9

    print("[TRAINING STARTED]")

    for epoch in range(300):

        model.train()
        train_loss = 0

        for bx, by in train_loader:

            bx = bx.to(DEVICE)
            by = by.to(DEVICE)

            optimizer.zero_grad()

            pred = model(bx)

            loss = physics_loss(pred, by, bx)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss += loss.item()

        ####################################
        # VALIDATION
        ####################################

        model.eval()

        with torch.no_grad():

            pred_val = model(X_val_tensor.to(DEVICE))

            val_loss = nn.MSELoss()(
                pred_val,
                Y_val_tensor.to(DEVICE)
            )

        if val_loss.item() < best_val:
            best_val = val_loss.item()

            torch.save(
                model.state_dict(),
                os.path.join(base_dir, "advanced_fusion_model.pth")
            )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | "
                f"Train: {train_loss/len(train_loader):.6f} | "
                f"Val: {val_loss.item():.6f}"
            )

    print("[DONE] Model saved.")


if __name__ == "__main__":

    base = r"C:\Users\furka\Documents\physical_informed_neural_network\data"

    train_advanced_model(base)
