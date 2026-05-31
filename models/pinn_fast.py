"""
PINN Hizli Versiyon - i5 CPU icin optimize
- 10 Hz (400Hz yerine) downsample
- Kucuk model
- 150 epoch
- Tahminen 10-15 dakika
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

DEVICE  = torch.device('cpu')
G       = 9.80665
OMEGA   = 7.2921150e-5
WGS84_A = 6378137.0
WGS84_F = 1/298.257223563
WGS84_E2= 2*WGS84_F - WGS84_F**2
DT_OUT  = 0.1    # 10 Hz
W       = 20     # 20 x 0.1s = 2 saniye pencere

IMU_PATH = r'C:\Users\furka\Documents\physical_informed_neural_network\bell412_dataset1_imu.csv'
MAG_PATH = r'C:\Users\furka\Documents\physical_informed_neural_network\bell412_dataset1_mag.csv'
PPK_PATH = r'C:\Users\furka\Documents\physical_informed_neural_network\data\dataset1\bell412_dataset1_ppk.pos'

print(f"Cihaz: CPU | {1/DT_OUT:.0f} Hz | Pencere: {W*DT_OUT:.1f}s")

def ppk_to_enu(ppk_path):
    rows = []
    with open(ppk_path) as f:
        for line in f:
            if line.startswith('%'): continue
            p = line.split()
            if len(p) < 6: continue
            try:
                rows.append([float(p[2]),float(p[3]),
                             float(p[4]),int(p[5])])
            except: pass
    arr = np.array(rows)
    lat_r=np.radians(arr[:,0]); lon_r=np.radians(arr[:,1])
    h=arr[:,2]; Q=arr[:,3]
    lat0,lon0,h0 = lat_r[0],lon_r[0],h[0]
    N  = WGS84_A/np.sqrt(1-WGS84_E2*np.sin(lat_r)**2)
    X  = (N+h)*np.cos(lat_r)*np.cos(lon_r)
    Y  = (N+h)*np.cos(lat_r)*np.sin(lon_r)
    Z  = (N*(1-WGS84_E2)+h)*np.sin(lat_r)
    N0 = WGS84_A/np.sqrt(1-WGS84_E2*np.sin(lat0)**2)
    X0 = (N0+h0)*np.cos(lat0)*np.cos(lon0)
    Y0 = (N0+h0)*np.cos(lat0)*np.sin(lon0)
    Z0 = (N0*(1-WGS84_E2)+h0)*np.sin(lat0)
    dX,dY,dZ = X-X0,Y-Y0,Z-Z0
    sl,cl=np.sin(lat0),np.cos(lat0)
    sn,cn=np.sin(lon0),np.cos(lon0)
    e=-sn*dX+cn*dY; n=-sl*cn*dX-sl*sn*dY+cl*dZ
    u=cl*cn*dX+cl*sn*dY+sl*dZ
    return np.column_stack([e,n,u]), Q, lat0

def load_data():
    print("\n=== VERİ HAZIRLANIYOR ===")
    imu = pd.read_csv(IMU_PATH).sort_values('time_s').reset_index(drop=True)
    t_imu = imu['time_s'].values
    fs = len(imu)/t_imu[-1]

    # NED -> ENU + yerc. cikarmasi
    ae = imu['acc_y'].values
    an = imu['acc_x'].values
    au = -imu['acc_z'].values - G

    # Dusuk gecis filtresi
    b,a = butter(4, 10.0/(fs/2), btype='low')
    ae_f=filtfilt(b,a,ae); an_f=filtfilt(b,a,an); au_f=filtfilt(b,a,au)
    gx_f=filtfilt(b,a,imu['gyro_x'].values)
    gy_f=filtfilt(b,a,imu['gyro_y'].values)
    gz_f=filtfilt(b,a,imu['gyro_z'].values)

    # 10 Hz downsample
    t_out = np.arange(0, t_imu[-1], DT_OUT)
    imu_ds = np.column_stack([
        interp1d(t_imu,ae_f)(t_out),
        interp1d(t_imu,an_f)(t_out),
        interp1d(t_imu,au_f)(t_out),
        interp1d(t_imu,gx_f)(t_out),
        interp1d(t_imu,gy_f)(t_out),
        interp1d(t_imu,gz_f)(t_out),
    ])
    print(f"IMU 10Hz: {len(imu_ds)} nokta | {t_out[-1]:.1f}s")

    # Manyetometre heading
    mag = pd.read_csv(MAG_PATH).sort_values('time_s').reset_index(drop=True)
    heading = np.arctan2(mag['mag_y'].values, mag['mag_x'].values)
    heading_ds = interp1d(mag['time_s'].values, heading,
                          bounds_error=False,
                          fill_value='extrapolate')(t_out)

    # PPK ground truth
    pos_enu, Q, lat0 = ppk_to_enu(PPK_PATH)
    t_ppk = np.linspace(0, t_out[-1], len(pos_enu))
    pos_ds = np.column_stack([
        interp1d(t_ppk,pos_enu[:,0],bounds_error=False,
                 fill_value='extrapolate')(t_out),
        interp1d(t_ppk,pos_enu[:,1],bounds_error=False,
                 fill_value='extrapolate')(t_out),
        interp1d(t_ppk,pos_enu[:,2],bounds_error=False,
                 fill_value='extrapolate')(t_out),
    ])

    print(f"ENU aralik:")
    print(f"  E:[{pos_ds[:,0].min():.1f},{pos_ds[:,0].max():.1f}]m")
    print(f"  N:[{pos_ds[:,1].min():.1f},{pos_ds[:,1].max():.1f}]m")
    print(f"  U:[{pos_ds[:,2].min():.1f},{pos_ds[:,2].max():.1f}]m")

    cor_f = 2*OMEGA*np.sin(lat0)
    return t_out, imu_ds, heading_ds, pos_ds, cor_f


class PINNNav(nn.Module):
    def __init__(self):
        super().__init__()
        # Kucuk model: GRU + basit head
        self.gru = nn.GRU(7, 128, num_layers=2,
                          batch_first=True, dropout=0.1)
        self.pos_enc = nn.Linear(3, 32)
        self.head = nn.Sequential(
            nn.Linear(160, 128), nn.Tanh(),
            nn.Linear(128, 6)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, seq, prev):
        _, h = self.gru(seq)
        h = h[-1]
        p = torch.tanh(self.pos_enc(prev))
        out = self.head(torch.cat([h,p],1))
        return prev + out[:,:3], out[:,3:]


def train(model, X_t, Xp_t, Y_t, M_t, cor_f, n_epochs=150, batch=128):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, 1e-5)
    mse = nn.MSELoss()
    n   = len(X_t)
    history = {'total':[],'data':[],'phys':[]}

    model.train()
    for ep in range(1, n_epochs+1):
        lam_p = min(0.05, (ep/100)*0.05)
        idx = torch.randperm(n)
        el=ed=ep_=0.0; nb=0

        for b in range(0,n,batch):
            bi = idx[b:b+batch]
            xi=X_t[bi]; xpi=Xp_t[bi]
            yi=Y_t[bi]; mi=M_t[bi]
            opt.zero_grad()
            pred,vel = model(xi,xpi)
            Ld = mse(pred,yi)
            if mi.sum()>0:
                Ld = Ld + mse(pred[mi],yi[mi])
            # Fizik: pos-hiz tutarliligi + hiz siniri
            dp = pred-xpi
            L1 = torch.mean((dp-vel*DT_OUT)**2)
            L2 = torch.mean(torch.relu(torch.norm(vel,dim=1)-80)**2)
            # Coriolis
            cx =  cor_f*vel[:,1]*DT_OUT
            cy = -cor_f*vel[:,0]*DT_OUT
            cor= torch.stack([cx,cy,torch.zeros(len(cx))],1)
            L3 = torch.mean((dp[:,:2]-vel[:,:2]*DT_OUT
                             -cor[:,:2])**2)*0.1
            loss = Ld + lam_p*(L1+L2+L3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            el+=loss.item(); ed+=Ld.item()
            ep_+=(L1+L2+L3).item(); nb+=1

        sch.step()
        history['total'].append(el/nb)
        history['data'].append(ed/nb)
        history['phys'].append(ep_/nb)

        # Her 10 epoch rapor
        if ep % 10 == 0:
            pct = ep/n_epochs*100
            bar = '█'*int(pct//5) + '░'*(20-int(pct//5))
            print(f"[{bar}] {ep:3d}/{n_epochs} | "
                  f"Veri:{ed/nb:.5f} | Fizik:{ep_/nb:.4f}",
                  flush=True)

    return history


def predict_seq(model, imu_ds, heading_ds, pos_ds, mask, sc_imu, sc_pos):
    imu_h = np.column_stack([imu_ds, heading_ds.reshape(-1,1)])
    imu_n = sc_imu.transform(imu_h)
    pos_n = sc_pos.transform(pos_ds)
    n = len(imu_ds)
    pos_pred = np.zeros((n,3)); pos_pred[0]=pos_ds[0]

    model.eval()
    with torch.no_grad():
        for i in range(W,n):
            win = torch.FloatTensor(imu_n[i-W:i]).unsqueeze(0)
            ps  = sc_pos.transform(pos_pred[i-1:i])
            pt  = torch.FloatTensor(ps)
            pred_s,_ = model(win,pt)
            pm = sc_pos.inverse_transform(pred_s.numpy())
            pos_pred[i] = pos_ds[i] if mask[i] else pm[0]
    return pos_pred


def plot_and_save(pos_pred, pos_gt, t, mask, history):
    err = np.sqrt(np.sum((pos_pred-pos_gt)**2,axis=1))
    out = np.where(~mask)[0]

    print(f"\n{'='*50}")
    print(f"ATE (tum):          {np.mean(err):.3f} m")
    print(f"ATE (GNSS kesinti): {np.mean(err[out]):.3f} m")
    print(f"RMSE E: {np.sqrt(np.mean((pos_pred[:,0]-pos_gt[:,0])**2)):.3f} m")
    print(f"RMSE N: {np.sqrt(np.mean((pos_pred[:,1]-pos_gt[:,1])**2)):.3f} m")
    print(f"RMSE U: {np.sqrt(np.mean((pos_pred[:,2]-pos_gt[:,2])**2)):.3f} m")
    print(f"Max:    {err.max():.3f} m")
    print(f"{'='*50}")

    fig = plt.figure(figsize=(18,10))
    ax1=fig.add_subplot(231,projection='3d')
    ax1.plot(*pos_gt.T,'g-',lw=2,label='GT(PPK)')
    ax1.plot(*pos_pred.T,'b--',lw=1.5,label='PINN',alpha=0.85)
    if len(out): ax1.plot(*pos_pred[out].T,'r-',lw=2,label='Kesinti')
    ax1.set_xlabel('E(m)');ax1.set_ylabel('N(m)');ax1.set_zlabel('U(m)')
    ax1.set_title('3B Yorunge');ax1.legend(fontsize=8)

    for k,(lbl,i) in enumerate([('Dogu E',0),('Kuzey N',1),('Yukseklik U',2)]):
        ax=fig.add_subplot(232+k)
        ax.plot(t,pos_gt[:,i],'g-',label='GT',alpha=0.85)
        ax.plot(t,pos_pred[:,i],'b--',label='PINN',alpha=0.85)
        if len(out):
            ax.axvspan(t[out[0]],t[min(out[-1],len(t)-1)],
                       alpha=0.15,color='red',label='Kesinti')
        ax.set_xlabel('Zaman(s)');ax.set_ylabel(f'{lbl}(m)')
        ax.set_title(lbl);ax.legend(fontsize=8)

    ax5=fig.add_subplot(235)
    ax5.plot(t,err,'r-',lw=1.5)
    ax5.axhline(np.mean(err),color='k',ls='--',
                label=f'ATE={np.mean(err):.2f}m')
    if len(out):
        ax5.axvspan(t[out[0]],t[min(out[-1],len(t)-1)],
                    alpha=0.15,color='red',label='Kesinti')
    ax5.set_xlabel('Zaman(s)');ax5.set_ylabel('Hata(m)')
    ax5.set_title('Pozisyon Hatasi');ax5.legend(fontsize=8)

    ax6=fig.add_subplot(236)
    ep=range(1,len(history['total'])+1)
    ax6.semilogy(ep,history['total'],'k-',lw=2,label='Toplam')
    ax6.semilogy(ep,history['data'],'b-',alpha=0.7,label='Veri')
    ax6.semilogy(ep,history['phys'],'r-',alpha=0.7,label='Fizik')
    ax6.set_xlabel('Epoch');ax6.set_ylabel('Kayip')
    ax6.set_title('Egitim');ax6.legend(fontsize=8)

    plt.suptitle(
        f'PINN 400Hz IMU+Mag | Bell412 | '
        f'ATE={np.mean(err):.2f}m | '
        f'Kesinti ATE={np.mean(err[out]):.2f}m',
        fontsize=12,fontweight='bold')
    plt.tight_layout()
    out_path = 'pinn_sonuclar.png'
    plt.savefig(out_path,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"Grafik kaydedildi: {out_path}")


def main():
    torch.manual_seed(42); np.random.seed(42)

    t, imu_ds, heading_ds, pos_ds, cor_f = load_data()
    n = len(t)

    # Kesinti: %35-55 arasi (~86s)
    mask = np.ones(n, dtype=bool)
    s,e  = int(n*0.35), int(n*0.55)
    mask[s:e] = False
    print(f"GNSS kesinti: {t[s]:.0f}s - {t[e]:.0f}s ({(e-s)*DT_OUT:.0f}s)")

    # Normalize
    sc_imu = StandardScaler()
    sc_pos = StandardScaler()
    imu_h  = np.column_stack([imu_ds, heading_ds.reshape(-1,1)])
    sc_imu.fit(imu_h); sc_pos.fit(pos_ds)
    imu_n = sc_imu.transform(imu_h)
    pos_n = sc_pos.transform(pos_ds)

    # Pencereler
    X_list, Xp_list, Y_list, M_list = [],[],[],[]
    for i in range(W,n):
        X_list.append(imu_n[i-W:i])
        Xp_list.append(pos_n[i-1])
        Y_list.append(pos_n[i])
        M_list.append(mask[i])

    X_t  = torch.FloatTensor(np.array(X_list))
    Xp_t = torch.FloatTensor(np.array(Xp_list))
    Y_t  = torch.FloatTensor(np.array(Y_list))
    M_t  = torch.BoolTensor(np.array(M_list))
    print(f"Pencere sayisi: {len(X_t)}")

    model = PINNNav()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} param\n")

    print("=== EGİTİM (150 epoch) ===\n")
    history = train(model, X_t, Xp_t, Y_t, M_t, cor_f,
                    n_epochs=150, batch=128)

    print("\n=== TEST ===")
    pos_pred = predict_seq(model, imu_ds, heading_ds,
                           pos_ds, mask, sc_imu, sc_pos)
    plot_and_save(pos_pred, pos_ds, t, mask, history)

    torch.save({'model':model.state_dict(),
                'sc_imu':sc_imu,'sc_pos':sc_pos},
               'pinn_model.pth')
    print("Model: pinn_model.pth")


if __name__ == '__main__':
    main()
