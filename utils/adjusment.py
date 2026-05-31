import numpy as np
import pandas as pd
import os
from scipy.linalg import lstsq


def process_block_adjustment(base_dir):
    print("[SYSTEM] Global Block Adjustment Started...\n")

    A_global = []
    b_global_x = []
    b_global_y = []

    for i in range(1, 7):
        file_path = os.path.join(
            base_dir,
            f"dataset{i}",
            f"dataset{i}_egitim_verisi.csv"
        )

        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        t = df['Zaman_sn'].values

        dt = np.diff(t, prepend=t[0])
        dt[0] = np.mean(dt[1:5]) if len(dt) > 5 else 0.02
        dt = np.clip(dt, 1e-4, None)

        # IMU velocity integration
        v_x_imu = np.cumsum(df['acc_x'].values * dt)
        v_y_imu = np.cumsum(df['acc_y'].values * dt)

        # Ground truth velocity
        v_x_true = np.gradient(df['pos_x_true'].values, t)
        v_y_true = np.gradient(df['pos_y_true'].values, t)

        # Residual
        v_diff_x = v_x_true - v_x_imu
        v_diff_y = v_y_true - v_y_imu

        # Design matrix
        A_local = np.column_stack([
            np.ones(len(df)),
            t
        ])

        A_global.append(A_local)
        b_global_x.append(v_diff_x)
        b_global_y.append(v_diff_y)

        print(f"[ADDED] Dataset {i} -> {len(df)} rows")

    A = np.vstack(A_global)
    bx = np.concatenate(b_global_x)
    by = np.concatenate(b_global_y)

    params_x, _, _, _ = lstsq(A, bx)
    params_y, _, _, _ = lstsq(A, by)

    print("\n==============================")
    print("IMU BIAS MODEL")
    print("==============================")
    print(f"X Bias Offset : {params_x[0]:.8f}")
    print(f"X Drift       : {params_x[1]:.8f}")
    print(f"Y Bias Offset : {params_y[0]:.8f}")
    print(f"Y Drift       : {params_y[1]:.8f}")
    print("==============================")

    return params_x, params_y


if __name__ == "__main__":
    base = r"C:\Users\furka\Documents\physical_informed_neural_network\data"
    process_block_adjustment(base)
