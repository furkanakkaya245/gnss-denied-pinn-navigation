"""
PINN Multi-Dataset Navigasyon
Egitim: DS1 + DS3 | Test: DS4
Bell 412, 400Hz IMU + Manyetometre + PPK
"""

import torch, torch.nn as nn
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import warnings; warnings.filterwarnings('ignore')
import time

DEVICE   = torch.device('cpu')
G        = 9.80665
OMEGA    = 7.2921150e-5
WGS84_A  = 6378137.0
WGS84_F  = 1/298.257223563
WGS84_E2 = 2*WGS84_F - WGS84_F**2
DT_OUT   = 0.1    # 10 Hz
W        = 20     # 2 saniye pencere

TRAIN_DS = [1, 3]
TEST_DS  = 4

IMU_PATHS = {i: f'C:\\Users\\furka\\Documents\\physical_informed_neural_network\\bell412_dataset{i}_imu.csv' for i in [1,3,4]}
MAG_PATHS = {i: f'C:\\Users\\furka\\Documents\\physical_informed_neural_network\\bell412_dataset{i}_mag.csv' for i in [1,3,4]}
PPK_PATHS = {i: f'C:\\Users\\furka\\Documents\\physical_informed_neural_network\\data\\dataset{i}\\bell412_dataset{i}_ppk.pos' for i in [1,3,4]}

# ─── WGS84 referans noktasi (tum datasetler icin ortak) ───────────────────────
LAT0_DEG, LON0_DEG, H0_M = 45.324219897, -75.664598745, 84.1594
LAT0, LON0 = np.radians(LAT0_DEG), np.radians(LON0_DEG)
COR_F = 2 * OMEGA * np.sin(LAT0)

print(f"Egitim: DS{TRAIN_DS} | Test: DS{TEST_DS}")
print(f"Referans: {LAT0_DEG}N, {LON0_DEG}E")
print(f"Coriolis: {COR_F:.6e} rad/s\n")


# ─── Koordinat donusumleri ────────────────────────────────────────────────────
def ppk_pos_to_enu(ppk_path):
    rows = []
    with open(ppk_path) as f:
        for line in f:
            if line.startswith('%'): continue
            p = line.split()
            if len(p) < 6: continue
            try:
                rows.append([float(p[2]), float(p[3]),
                              float(p[4]), int(p[5])])
            except: pass
    arr = np.array(rows)
    lr = np.radians(arr[:,0]); lo = np.radians(arr[:,1])
    h = arr[:,2]

    N0 = WGS84_A/np.sqrt(1-WGS84_E2*np.sin(LAT0)**2)
    X0 = (N0+H0_M)*np.cos(LAT0)*np.cos(LON0)
    Y0 = (N0+H0_M)*np.cos(LAT0)*np.sin(LON0)
    Z0 = (N0*(1-WGS84_E2)+H0_M)*np.sin(LAT0)

    N  = WGS84_A/np.sqrt(1-WGS84_E2*np.sin(lr)**2)
    X  = (N+h)*np.cos(lr)*np.cos(lo)
    Y  = (N+h)*np.cos(lr)*np.sin(lo)
    Z  = (N*(1-WGS84_E2)+h)*np.sin(lr)

    dX,dY,dZ = X-X0, Y-Y0, Z-Z0
    sl,cl = np.sin(LAT0),np.cos(LAT0)
    sn,cn = np.sin(LON0),np.cos(LON0)
    e =  -sn*dX + cn*dY
    n =  -sl*cn*dX - sl*sn*dY + cl*dZ
    u =   cl*cn*dX + cl*sn*dY + sl*dZ
    return np.column_stack([e,n,u])


# ─── Tek dataset yukle ve isle ────────────────────────────────────────────────
def load_dataset(ds_id, outage_ratio=0.25):
    imu = pd.read_csv(IMU_PATHS[ds_id]).sort_values('time_s').reset_index(drop=True)
    mag = pd.read_csv(MAG_PATHS[ds_id]).sort_values('time_s').reset_index(drop=True)
    t_imu = imu['time_s'].values
    fs = len(imu)/t_imu[-1]

    # NED -> ENU + yerc. cikarmasi
    ae = imu['acc_y'].values
    an = imu['acc_x'].values
    au = -imu['acc_z'].values - G

    # Alcak gecis filtresi
    b,a = butter(4, 10.0/(fs/2), btype='low')
    channels = [ae, an, au,
                imu['gyro_x'].values,
                imu['gyro_y'].values,
                imu['gyro_z'].values]
    filtered = [filtfilt(b,a,ch) for ch in channels]

    # 10 Hz downsample
    t_out = np.arange(0, t_imu[-1], DT_OUT)
    imu_ds = np.column_stack([
        interp1d(t_imu, ch)(t_out) for ch in filtered])

    # Manyetometre heading
    t_mag = mag['time_s'].values
    heading = np.arctan2(mag['mag_y'].values, mag['mag_x'].values)
    heading_ds = interp1d(t_mag, heading,
                          bounds_error=False,
                          fill_value='extrapolate')(t_out)

    # PPK ENU (ortak referans)
    pos_enu = ppk_pos_to_enu(PPK_PATHS[ds_id])
    n_ppk = len(pos_enu)
    t_ppk = np.linspace(0, t_out[-1], n_ppk)
    pos_ds = np.column_stack([
        interp1d(t_ppk, pos_enu[:,k],
                 bounds_error=False,
                 fill_value='extrapolate')(t_out)
        for k in range(3)])

    # GNSS kesinti maskesi
    n = len(t_out)
    mask = np.ones(n, dtype=bool)
    s = int(n*0.35); e = int(n*(0.35+outage_ratio))
    mask[s:e] = False

    print(f"  DS{ds_id}: {n} nokta | {t_out[-1]:.0f}s | "
          f"Kesinti: {t_out[s]:.0f}-{t_out[e]:.0f}s ({(e-s)*DT_OUT:.0f}s) | "
          f"ENU E=[{pos_ds[:,0].min():.0f},{pos_ds[:,0].max():.0f}]m")

    return t_out, imu_ds, heading_ds, pos_ds, mask


# ─── Model ───────────────────────────────────────────────────────────────────
class PINNNav(nn.Module):
    def __init__(self, W=W, hidden=128):
        super().__init__()
        self.gru = nn.GRU(7, hidden, num_layers=2,
                          batch_first=True, dropout=0.1)
        self.pos_enc = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(hidden+32, 128), nn.Tanh(),
            nn.Linear(128, 6))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, seq, prev):
        _, h = self.gru(seq); h = h[-1]
        p = self.pos_enc(prev)
        out = self.head(torch.cat([h,p],1))
        return prev + out[:,:3], out[:,3:]


# ─── Fizik kaybi ─────────────────────────────────────────────────────────────
def phys_loss(prev, next_p, vel, dt=DT_OUT):
    dp = next_p - prev
    L1 = torch.mean((dp - vel*dt)**2)
    L2 = torch.mean(torch.relu(torch.norm(vel,dim=1)-80)**2)
    cx =  COR_F*vel[:,1]*dt
    cy = -COR_F*vel[:,0]*dt
    cor = torch.stack([cx,cy,torch.zeros(len(cx))],1)
    L3 = torch.mean((dp[:,:2]-vel[:,:2]*dt-cor[:,:2])**2)*0.1
    return L1, L2, L3


# ─── Egitim ──────────────────────────────────────────────────────────────────
def train(model, datasets, sc_imu, sc_pos,
          n_epochs=200, batch=128):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, 1e-5)
    mse = nn.MSELoss()
    history = {'total':[],'data':[],'phys':[]}

    # Tum egitim pencerelerini hazirla
    all_X, all_Xp, all_Y, all_M = [], [], [], []
    for (t, imu_ds, heading_ds, pos_ds, mask) in datasets:
        n = len(t)
        imu_h = np.column_stack([imu_ds, heading_ds.reshape(-1,1)])
        imu_n = sc_imu.transform(imu_h)
        pos_n = sc_pos.transform(pos_ds)
        for i in range(W, n):
            all_X.append(imu_n[i-W:i])
            all_Xp.append(pos_n[i-1])
            all_Y.append(pos_n[i])
            all_M.append(mask[i])

    X_t  = torch.FloatTensor(np.array(all_X))
    Xp_t = torch.FloatTensor(np.array(all_Xp))
    Y_t  = torch.FloatTensor(np.array(all_Y))
    M_t  = torch.BoolTensor(np.array(all_M))
    print(f"  Toplam pencere: {len(X_t)} "
          f"({len(X_t)*DT_OUT:.0f}s egitim verisi)")

    model.train()
    t0 = time.time()
    for ep in range(1, n_epochs+1):
        lam_p = min(0.05, (ep/100)*0.05)
        idx = torch.randperm(len(X_t))
        el=ed=ep_=0.0; nb=0

        for b in range(0, len(X_t), batch):
            bi = idx[b:b+batch]
            xi=X_t[bi]; xpi=Xp_t[bi]
            yi=Y_t[bi]; mi=M_t[bi]
            opt.zero_grad()
            pred, vel = model(xi, xpi)
            Ld = mse(pred, yi)
            if mi.sum() > 0:
                Ld = Ld + mse(pred[mi], yi[mi])
            L1,L2,L3 = phys_loss(xpi, pred, vel)
            loss = Ld + lam_p*(L1+L2+L3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el+=loss.item(); ed+=Ld.item()
            ep_+=(L1+L2+L3).item(); nb+=1

        sch.step()
        history['total'].append(el/nb)
        history['data'].append(ed/nb)
        history['phys'].append(ep_/nb)

        if ep % 10 == 0:
            elapsed = time.time()-t0
            remaining = elapsed/ep*(n_epochs-ep)
            pct = ep/n_epochs*100
            bar = '█'*int(pct//5)+'░'*(20-int(pct//5))
            print(f"[{bar}] {ep:3d}/{n_epochs} | "
                  f"Veri:{ed/nb:.5f} | Fizik:{ep_/nb:.5f} | "
                  f"Kalan:{remaining/60:.1f}dk", flush=True)

    return history


# ─── Ardisik tahmin ───────────────────────────────────────────────────────────
def predict(model, t, imu_ds, heading_ds, pos_ds, mask, sc_imu, sc_pos):
    imu_h = np.column_stack([imu_ds, heading_ds.reshape(-1,1)])
    imu_n = sc_imu.transform(imu_h)
    pos_n = sc_pos.transform(pos_ds)
    n = len(t)
    pos_pred = np.zeros((n,3)); pos_pred[0] = pos_ds[0]

    model.eval()
    with torch.no_grad():
        for i in range(W, n):
            win = torch.FloatTensor(imu_n[i-W:i]).unsqueeze(0)
            ps  = sc_pos.transform(pos_pred[i-1:i])
            pt  = torch.FloatTensor(ps)
            pred_s, _ = model(win, pt)
            pm = sc_pos.inverse_transform(pred_s.numpy())
            pos_pred[i] = pos_ds[i] if mask[i] else pm[0]
    return pos_pred


# ─── Degerlendirme ────────────────────────────────────────────────────────────
def evaluate(pos_pred, pos_gt, mask, label="DS"):
    err = np.sqrt(np.sum((pos_pred-pos_gt)**2, axis=1))
    out = np.where(~mask)[0]
    ate  = np.mean(err)
    ate_o= np.mean(err[out]) if len(out) else 0
    rmse_e = np.sqrt(np.mean((pos_pred[:,0]-pos_gt[:,0])**2))
    rmse_n = np.sqrt(np.mean((pos_pred[:,1]-pos_gt[:,1])**2))
    rmse_u = np.sqrt(np.mean((pos_pred[:,2]-pos_gt[:,2])**2))

    print(f"\n{'='*55}")
    print(f"TEST SONUCLARI — {label}")
    print(f"{'='*55}")
    print(f"ATE (tum yörünge):    {ate:.3f} m")
    print(f"ATE (GNSS kesinti):   {ate_o:.3f} m")
    print(f"RMSE Dogu (E):        {rmse_e:.3f} m")
    print(f"RMSE Kuzey (N):       {rmse_n:.3f} m")
    print(f"RMSE Yukseklik (U):   {rmse_u:.3f} m")
    print(f"Maks hata:            {err.max():.3f} m")
    print(f"{'='*55}")
    return err, out


# ─── Grafik ──────────────────────────────────────────────────────────────────
def plot(pos_pred, pos_gt, err, out_idx, t, history, ds_id, ate, ate_out):
    fig = plt.figure(figsize=(20,12))

    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(*pos_gt.T,   'g-',  lw=2,   label='GT (PPK/ENU)')
    ax1.plot(*pos_pred.T, 'b--', lw=1.5, label='PINN', alpha=0.85)
    if len(out_idx):
        ax1.plot(*pos_pred[out_idx].T,'r-',lw=2.5,label='GNSS Kesinti')
    ax1.set_xlabel('E(m)');ax1.set_ylabel('N(m)');ax1.set_zlabel('U(m)')
    ax1.set_title(f'3B ENU Yorunge — DS{ds_id}')
    ax1.legend(fontsize=8)

    for k,(lbl,i) in enumerate([('Dogu E',0),('Kuzey N',1),('Yukseklik U',2)]):
        ax = fig.add_subplot(232+k)
        ax.plot(t, pos_gt[:,i],   'g-',  label='GT',   alpha=0.85)
        ax.plot(t, pos_pred[:,i], 'b--', label='PINN', alpha=0.85)
        if len(out_idx):
            ax.axvspan(t[out_idx[0]], t[min(out_idx[-1],len(t)-1)],
                       alpha=0.15, color='red', label='Kesinti')
        ax.set_xlabel('Zaman(s)'); ax.set_ylabel(f'{lbl}(m)')
        ax.set_title(lbl); ax.legend(fontsize=8)

    ax5 = fig.add_subplot(235)
    ax5.plot(t, err, 'r-', lw=1.5)
    ax5.axhline(ate, color='k', ls='--', label=f'ATE={ate:.2f}m')
    if len(out_idx):
        ax5.axvspan(t[out_idx[0]], t[min(out_idx[-1],len(t)-1)],
                    alpha=0.15, color='red', label=f'Kesinti ATE={ate_out:.2f}m')
    ax5.set_xlabel('Zaman(s)'); ax5.set_ylabel('Hata(m)')
    ax5.set_title('Pozisyon Hatasi'); ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(236)
    ep = range(1,len(history['total'])+1)
    ax6.semilogy(ep,history['total'],'k-',lw=2,label='Toplam')
    ax6.semilogy(ep,history['data'],'b-',alpha=0.7,label='Veri')
    ax6.semilogy(ep,history['phys'],'r-',alpha=0.7,label='Fizik')
    ax6.set_xlabel('Epoch');ax6.set_ylabel('Kayip(log)')
    ax6.set_title('Egitim Kaybi');ax6.legend(fontsize=8)

    plt.suptitle(
        f'PINN Multi-Dataset (DS1+DS3→DS4) | ENU Ortak Referans | '
        f'ATE={ate:.2f}m | Kesinti ATE={ate_out:.2f}m',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = 'pinn_multi_sonuclar.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafik: {out}")


# ─── Ana akis ────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42); np.random.seed(42)

    print("=== VERİ YÜKLENİYOR ===\n")
    train_data = []
    for ds in TRAIN_DS:
        d = load_dataset(ds, outage_ratio=0.25)
        train_data.append(d)

    print()
    test_data = load_dataset(TEST_DS, outage_ratio=0.25)
    t_tst, imu_tst, head_tst, pos_tst, mask_tst = test_data

    # Global scaler — sadece egitim verisiyle fit
    all_imu, all_pos = [], []
    for (t,imu_ds,heading_ds,pos_ds,mask) in train_data:
        imu_h = np.column_stack([imu_ds, heading_ds.reshape(-1,1)])
        all_imu.append(imu_h)
        all_pos.append(pos_ds)

    sc_imu = StandardScaler().fit(np.vstack(all_imu))
    sc_pos = StandardScaler().fit(np.vstack(all_pos))

    model = PINNNav(W=W, hidden=128)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} param")

    print(f"\n=== EGİTİM (DS{TRAIN_DS}) ===\n")
    history = train(model, train_data, sc_imu, sc_pos,
                    n_epochs=200, batch=128)

    print(f"\n=== TEST (DS{TEST_DS}) ===")
    pos_pred = predict(model, t_tst, imu_tst, head_tst,
                       pos_tst, mask_tst, sc_imu, sc_pos)
    err, out_idx = evaluate(pos_pred, pos_tst, mask_tst,
                            label=f"DS{TEST_DS}")

    ate     = np.mean(err)
    ate_out = np.mean(err[out_idx]) if len(out_idx) else 0
    plot(pos_pred, pos_tst, err, out_idx, t_tst,
         history, TEST_DS, ate, ate_out)

    torch.save({'model':model.state_dict(),
                'sc_imu':sc_imu,'sc_pos':sc_pos},
               'pinn_multi_model.pth')
    print("Model: pinn_multi_model.pth")


if __name__ == '__main__':
    main()
