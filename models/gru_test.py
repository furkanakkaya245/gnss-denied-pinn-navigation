import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# --- MODEL MİMARİSİ ---
class GRUFusionNet(nn.Module):
    def __init__(self):
        super(GRUFusionNet, self).__init__()
        # 6 özellik girişiyle uyumlu
        self.gru = nn.GRU(input_size=6, hidden_size=128, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- DÜZELTİLMİŞ FONKSİYON İSMİ ---
def create_sequences(X, seq_length=15):
    xs = []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i+seq_length)])
    return np.array(xs)

def test_pro(base_dir, dataset_no):
    # Modeli Yükle
    model = GRUFusionNet()
    model.load_state_dict(torch.load(os.path.join(base_dir, "bell412_pro_model.pth")))
    model.eval()
    
    # Veriyi Hazırla
    df = pd.read_csv(os.path.join(base_dir, f"dataset{dataset_no}", f"dataset{dataset_no}_egitim_verisi.csv"))
    t = df['Zaman_sn'].values
    df['acc_x_clean'] = df['acc_x'] - (0.00094616 * t - 0.10833055)
    df['acc_y_clean'] = df['acc_y'] - (-0.00074795 * t - 0.10677880)
    
    features = ['acc_x_clean', 'acc_y_clean', 'gyro_z', 'gnss_x', 'gnss_y', 'gnss_Q']
    
    # Testte GNSS'i KESİNTİYE UĞRAT (Simülasyon)
    df_sim = df.copy()
    mask = (df_sim['Zaman_sn'] >= 100) & (df_sim['Zaman_sn'] <= 400)
    df_sim.loc[mask, ['gnss_x', 'gnss_y', 'gnss_Q']] = 0
    
    # Scaler'ı orijinal eğitim verisiyle eğitmiştik, burada da fit_transform
    X = StandardScaler().fit_transform(df_sim[features].values)
    
    # Düzeltilmiş Çağrı
    X_seq = torch.tensor(create_sequences(X), dtype=torch.float32)
    
    # Tahmin
    with torch.no_grad():
        delta_pred = model(X_seq).numpy()
    
    # Hata analizi (Hizalama: 15 kayma var)
    pred_x = df['gnss_x'].iloc[15:].values + delta_pred[:, 0]
    pred_y = df['gnss_y'].iloc[15:].values + delta_pred[:, 1]
    
    mask_idx = (df['Zaman_sn'].iloc[15:] >= 100) & (df['Zaman_sn'].iloc[15:] <= 400)
    rmse = np.sqrt(np.mean((pred_x[mask_idx] - df['pos_x_true'].iloc[15:][mask_idx])**2 + 
                           (pred_y[mask_idx] - df['pos_y_true'].iloc[15:][mask_idx])**2))
    
    print(f"=== GERÇEK TEST SONUCU ===")
    print(f"Kör Uçuş (100-400s) RMSE: {rmse:.4f} Metre")
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['pos_x_true'], df['pos_y_true'], 'g', label='Gerçek')
    plt.plot(pred_x, pred_y, 'b--', label='Pro GRU AI')
    plt.legend(); plt.show()

if __name__ == "__main__":
    base = r"C:\Users\furka\Documents\physical_informed_neural_network\data"
    test_pro(base, 5)