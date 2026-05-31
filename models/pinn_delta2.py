"""
PINN Delta Pozisyon Yaklaşımı v2
Egitim: DS1 + DS3 | Test: DS4

Temel degisiklik:
- Model mutlak konum degil DELTA pozisyon ogrenir
- Girdi: IMU penceresi (koordinat bagimsiz)
- Hedef: pos[i] - pos[i-1] (adim adim yer degistirme)
- Test: delta tahminleri kumulatif toplanarak yörünge olusturulur
- Bu yaklasim farkli yörüngelere genellenir
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
DT_OUT   = 0.1
W        = 20     # 2 saniye pencere

TRAIN_DS = [1, 3]
TEST_DS  = 4

IMU_PATHS = {i: f'C:\\Users\\furka\\Documents\\physical_informed_neural_network\\bell412_dataset{i}_imu.csv' for i in [1,3,4]}
MAG_PATHS = {i: f'C:\\Users\\furka\\Documents\\physical_informed_neural_network\\bell412_dataset{i}_mag.csv' for i in [1,3,4]}
PPK_PATHS = {i: f'C:\\Users\\furka\\Documents\\physical_informed_neural_network\\data\\dataset{i}\\bell412_dataset{i}_ppk.pos' for i in [1,3,4]}

LAT0_DEG, LON0_DEG, H0_M = 45.324219897, -75.664598745, 84.1594
LAT0, LON0 = np.radians(LAT0_DEG), np.radians(LON0_DEG)
COR_F = 2 * OMEGA * np.sin(LAT0)

print(f"Egitim: DS{TRAIN_DS} | Test: DS{TEST_DS}")
print(f"Delta pozisyon yaklasimi — koordinat bagimsiz\n")


# ─── Koordinat donusumleri ────────────────────────────────────────────────────
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
    lr=np.radians(arr[:,0]); lo=np.radians(arr[:,1]); h=arr[:,2]

    N0 = WGS84_A/np.sqrt(1-WGS84_E2*np.sin(LAT0)**2)
    X0=(N0+H0_M)*np.cos(LAT0)*np.cos(LON0)
    Y0=(N0+H0_M)*np.cos(LAT0)*np.sin(LON0)
    Z0=(N0*(1-WGS84_E2)+H0_M)*np.sin(LAT0)

    N =WGS84_A/np.sqrt(1-WGS84_E2*np.sin(lr)**2)
    X =(N+h)*np.cos(lr)*np.cos(lo)
    Y =(N+h)*np.cos(lr)*np.sin(lo)
    Z =(N*(1-WGS84_E2)+h)*np.sin(lr)

    dX,dY,dZ=X-X0,Y-Y0,Z-Z0
    sl,cl=np.sin(LAT0),np.cos(LAT0)
    sn,cn=np.sin(LON0),np.cos(LON0)
    e=-sn*dX+cn*dY
    n=-sl*cn*dX-sl*sn*dY+cl*dZ
    u=cl*cn*dX+cl*sn*dY+sl*dZ
    return np.column_stack([e,n,u])


# ─── Veri yukleme ─────────────────────────────────────────────────────────────
def load_dataset(ds_id, outage_ratio=0.25):
    imu=pd.read_csv(IMU_PATHS[ds_id]).sort_values('time_s').reset_index(drop=True)
    mag=pd.read_csv(MAG_PATHS[ds_id]).sort_values('time_s').reset_index(drop=True)
    t_imu=imu['time_s'].values; fs=len(imu)/t_imu[-1]

    # NED->ENU + yerc. cikarmasi
    ae=imu['acc_y'].values; an=imu['acc_x'].values
    au=-imu['acc_z'].values-G
    b,a=butter(4,10.0/(fs/2),btype='low')
    chs=[ae,an,au,imu['gyro_x'].values,
         imu['gyro_y'].values,imu['gyro_z'].values]
    flt=[filtfilt(b,a,ch) for ch in chs]

    t_out=np.arange(0,t_imu[-1],DT_OUT)
    imu_ds=np.column_stack([interp1d(t_imu,ch)(t_out) for ch in flt])

    # Manyetometre heading
    t_mag=mag['time_s'].values
    heading=np.arctan2(mag['mag_y'].values,mag['mag_x'].values)
    heading_ds=interp1d(t_mag,heading,
                        bounds_error=False,
                        fill_value='extrapolate')(t_out)

    # PPK ENU
    pos_enu=ppk_to_enu(PPK_PATHS[ds_id])
    t_ppk=np.linspace(0,t_out[-1],len(pos_enu))
    pos_ds=np.column_stack([
        interp1d(t_ppk,pos_enu[:,k],
                 bounds_error=False,
                 fill_value='extrapolate')(t_out)
        for k in range(3)])

    # Delta pozisyon (ground truth)
    delta_gt=np.diff(pos_ds,axis=0)   # [N-1, 3]

    # GNSS kesinti maskesi
    n=len(t_out)
    mask=np.ones(n,dtype=bool)
    s=int(n*0.35); e=int(n*(0.35+outage_ratio))
    mask[s:e]=False

    print(f"  DS{ds_id}: {n} nokta | {t_out[-1]:.0f}s | "
          f"Kesinti: {t_out[s]:.0f}-{t_out[e]:.0f}s | "
          f"delta_gt ort: {np.abs(delta_gt).mean():.4f}m/adim")

    return t_out, imu_ds, heading_ds, pos_ds, delta_gt, mask


# ─── Model: Delta tahmincisi ──────────────────────────────────────────────────
class PINNDeltaNet(nn.Module):
    """
    Girdi : IMU penceresi [B, W, 7] — acc(3)+gyro(3)+heading(1)
    Cikti : delta_pos [B, 3] + vel_est [B, 3]

    prev_pos YOK — koordinat bagimsiz.
    Model sadece IMU dinamiklerinden hareket vektoru ogrenir.
    """
    def __init__(self, W=W, hidden=128):
        super().__init__()
        # CNN: yerel IMU pattern tespiti
        self.cnn = nn.Sequential(
            nn.Conv1d(7, 64, 3, padding=1), nn.Tanh(),
            nn.Conv1d(64, 64, 3, padding=1), nn.Tanh(),
        )
        # GRU: zaman bagimli dinamikler
        self.gru = nn.GRU(64, hidden, num_layers=2,
                          batch_first=True, dropout=0.1)
        # Cikti kafasi
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.Tanh(),
            nn.Linear(128, 64),    nn.Tanh(),
            nn.Linear(64, 6)       # delta_pos(3) + vel_est(3)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, seq):
        # seq: [B, W, 7]
        x=self.cnn(seq.permute(0,2,1)).permute(0,2,1)  # [B,W,64]
        _,h=self.gru(x); h=h[-1]                        # [B,hidden]
        out=self.head(h)                                 # [B,6]
        return out[:,:3], out[:,3:]  # delta_pos, vel_est


# ─── Fizik kaybi ─────────────────────────────────────────────────────────────
def phys_loss(delta_pred, vel_est, imu_win, dt=DT_OUT):
    # L1: delta/dt = vel tutarliligi
    L1 = torch.mean((delta_pred - vel_est*dt)**2)

    # L2: hiz siniri (heli <80 m/s)
    v_norm = torch.norm(vel_est, dim=1)
    L2 = torch.mean(torch.relu(v_norm - 80.0)**2)

    # L3: Newton — son IMU adiminin ivmesiyle hiz degisimi tutarli olmali
    # dv ~ a_son * dt
    a_son = imu_win[:, -1, :3]  # son adim ivmesi [B,3]
    dv_expected = a_son * dt
    # vel_est ~ onceki hiz + ivme*dt (yaklasim)
    L3 = torch.mean((vel_est - dv_expected)**2) * 0.01

    # L4: Coriolis (yatay)
    cx =  COR_F * vel_est[:,1] * dt
    cy = -COR_F * vel_est[:,0] * dt
    cor = torch.stack([cx, cy, torch.zeros(len(cx))], 1)
    L4 = torch.mean((delta_pred - vel_est*dt - cor)**2) * 0.01

    return L1, L2, L3, L4


# ─── Egitim ──────────────────────────────────────────────────────────────────
def train(model, datasets, sc_imu, sc_delta,
          n_epochs=200, batch=128):
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,n_epochs,1e-5)
    mse=nn.MSELoss()
    history={'total':[],'data':[],'phys':[]}

    # Tum egitim pencerelerini hazirla
    all_X, all_Y = [], []
    for (t,imu_ds,heading_ds,pos_ds,delta_gt,mask) in datasets:
        n=len(t)
        imu_h=np.column_stack([imu_ds,heading_ds.reshape(-1,1)])
        imu_n=sc_imu.transform(imu_h)
        delta_n=sc_delta.transform(delta_gt)
        for i in range(W, n-1):
            all_X.append(imu_n[i-W:i])
            all_Y.append(delta_n[i])   # delta[i] = pos[i+1]-pos[i]

    X_t=torch.FloatTensor(np.array(all_X))
    Y_t=torch.FloatTensor(np.array(all_Y))
    print(f"  Toplam pencere: {len(X_t)}")

    model.train()
    t0=time.time()
    for ep in range(1,n_epochs+1):
        lam_p=min(0.1,(ep/100)*0.1)
        idx=torch.randperm(len(X_t))
        el=ed=ep_=0.0; nb=0

        for b in range(0,len(X_t),batch):
            bi=idx[b:b+batch]
            xi=X_t[bi]; yi=Y_t[bi]
            opt.zero_grad()
            delta_pred,vel_est=model(xi)
            Ld=mse(delta_pred,yi)
            L1,L2,L3,L4=phys_loss(delta_pred,vel_est,xi)
            loss=Ld+lam_p*(L1+L2+L3+L4)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            el+=loss.item(); ed+=Ld.item()
            ep_+=(L1+L2+L3+L4).item(); nb+=1

        sch.step()
        history['total'].append(el/nb)
        history['data'].append(ed/nb)
        history['phys'].append(ep_/nb)

        if ep%10==0:
            elapsed=time.time()-t0
            rem=elapsed/ep*(n_epochs-ep)
            pct=ep/n_epochs*100
            bar='█'*int(pct//5)+'░'*(20-int(pct//5))
            print(f"[{bar}] {ep:3d}/{n_epochs} | "
                  f"Veri:{ed/nb:.6f} | Fizik:{ep_/nb:.6f} | "
                  f"Kalan:{rem/60:.1f}dk",flush=True)

    return history


# ─── Test: Kumulatif delta tahmini ────────────────────────────────────────────
def predict_cumulative(model, t, imu_ds, heading_ds,
                       pos_ds, delta_gt, mask, sc_imu, sc_delta):
    """
    GNSS varsa: ground truth pos kullan (referans kaybi yok)
    GNSS yoksa: son bilinen konumdan delta tahminlerini biriktir
    """
    imu_h=np.column_stack([imu_ds,heading_ds.reshape(-1,1)])
    imu_n=sc_imu.transform(imu_h)
    n=len(t)
    pos_pred=np.zeros((n,3))
    pos_pred[0]=pos_ds[0]

    model.eval()
    with torch.no_grad():
        for i in range(W,n-1):
            if mask[i]:
                # GNSS var: ground truth kullan
                pos_pred[i]=pos_ds[i]
            else:
                # GNSS yok: delta tahmin et, biraktir
                win=torch.FloatTensor(imu_n[i-W:i]).unsqueeze(0)
                delta_s,_=model(win)
                delta_m=sc_delta.inverse_transform(delta_s.numpy())
                pos_pred[i]=pos_pred[i-1]+delta_m[0]

    pos_pred[-1]=pos_ds[-1] if mask[-1] else pos_pred[-2]
    return pos_pred


# ─── Degerlendirme ────────────────────────────────────────────────────────────
def evaluate(pos_pred, pos_gt, mask, label="DS"):
    err=np.sqrt(np.sum((pos_pred-pos_gt)**2,axis=1))
    out=np.where(~mask)[0]
    ate=np.mean(err)
    ate_o=np.mean(err[out]) if len(out) else 0

    print(f"\n{'='*55}")
    print(f"TEST — {label}")
    print(f"{'='*55}")
    print(f"ATE (tum):            {ate:.3f} m")
    print(f"ATE (GNSS kesinti):   {ate_o:.3f} m")
    print(f"RMSE E:               {np.sqrt(np.mean((pos_pred[:,0]-pos_gt[:,0])**2)):.3f} m")
    print(f"RMSE N:               {np.sqrt(np.mean((pos_pred[:,1]-pos_gt[:,1])**2)):.3f} m")
    print(f"RMSE U:               {np.sqrt(np.mean((pos_pred[:,2]-pos_gt[:,2])**2)):.3f} m")
    print(f"Maks hata:            {err.max():.3f} m")
    print(f"{'='*55}")
    return err, out


# ─── Grafik ──────────────────────────────────────────────────────────────────
def plot(pos_pred,pos_gt,err,out_idx,t,history,ds_id,ate,ate_out):
    fig=plt.figure(figsize=(20,12))

    ax1=fig.add_subplot(231,projection='3d')
    ax1.plot(*pos_gt.T,'g-',lw=2,label='GT (PPK/ENU)')
    ax1.plot(*pos_pred.T,'b--',lw=1.5,label='PINN',alpha=0.85)
    if len(out_idx):
        ax1.plot(*pos_pred[out_idx].T,'r-',lw=2.5,label='GNSS Kesinti')
    ax1.set_xlabel('E(m)');ax1.set_ylabel('N(m)');ax1.set_zlabel('U(m)')
    ax1.set_title(f'3B ENU Yorunge — DS{ds_id}')
    ax1.legend(fontsize=8)

    for k,(lbl,i) in enumerate([('Dogu E',0),('Kuzey N',1),('Yukseklik U',2)]):
        ax=fig.add_subplot(232+k)
        ax.plot(t,pos_gt[:,i],'g-',label='GT',alpha=0.85)
        ax.plot(t,pos_pred[:,i],'b--',label='PINN',alpha=0.85)
        if len(out_idx):
            ax.axvspan(t[out_idx[0]],t[min(out_idx[-1],len(t)-1)],
                       alpha=0.15,color='red',label='Kesinti')
        ax.set_xlabel('Zaman(s)');ax.set_ylabel(f'{lbl}(m)')
        ax.set_title(lbl);ax.legend(fontsize=8)

    ax5=fig.add_subplot(235)
    ax5.plot(t,err,'r-',lw=1.5)
    ax5.axhline(ate,color='k',ls='--',label=f'ATE={ate:.2f}m')
    if len(out_idx):
        ax5.axvspan(t[out_idx[0]],t[min(out_idx[-1],len(t)-1)],
                    alpha=0.15,color='red',
                    label=f'Kesinti ATE={ate_out:.2f}m')
    ax5.set_xlabel('Zaman(s)');ax5.set_ylabel('Hata(m)')
    ax5.set_title('Pozisyon Hatasi');ax5.legend(fontsize=8)

    ax6=fig.add_subplot(236)
    ep=range(1,len(history['total'])+1)
    ax6.semilogy(ep,history['total'],'k-',lw=2,label='Toplam')
    ax6.semilogy(ep,history['data'],'b-',alpha=0.7,label='Veri')
    ax6.semilogy(ep,history['phys'],'r-',alpha=0.7,label='Fizik')
    ax6.set_xlabel('Epoch');ax6.set_ylabel('Kayip(log)')
    ax6.set_title('Egitim');ax6.legend(fontsize=8)

    plt.suptitle(
        f'PINN Delta-Nav (DS1+DS3→DS4) | '
        f'ATE={ate:.2f}m | Kesinti ATE={ate_out:.2f}m',
        fontsize=11,fontweight='bold')
    plt.tight_layout()
    out='pinn_delta_sonuclar.png'
    plt.savefig(out,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"Grafik: {out}")


# ─── Ana akis ────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42); np.random.seed(42)

    print("=== VERİ YÜKLENİYOR ===\n")
    train_data=[]
    for ds in TRAIN_DS:
        d=load_dataset(ds)
        train_data.append(d)

    print()
    test_data=load_dataset(TEST_DS)
    t_t,imu_t,head_t,pos_t,delta_t,mask_t=test_data

    # Global scaler — sadece egitim verisi
    all_imu,all_delta=[],[]
    for (_,imu_ds,hds,pos_ds,delta_gt,_) in train_data:
        imu_h=np.column_stack([imu_ds,hds.reshape(-1,1)])
        all_imu.append(imu_h)
        all_delta.append(delta_gt)

    sc_imu  =StandardScaler().fit(np.vstack(all_imu))
    sc_delta=StandardScaler().fit(np.vstack(all_delta))

    # Delta istatistikleri
    all_d=np.vstack(all_delta)
    print(f"\nDelta GT istatistikleri (egitim):")
    print(f"  E: ort={all_d[:,0].mean():.4f} std={all_d[:,0].std():.4f} m")
    print(f"  N: ort={all_d[:,1].mean():.4f} std={all_d[:,1].std():.4f} m")
    print(f"  U: ort={all_d[:,2].mean():.4f} std={all_d[:,2].std():.4f} m")

    model=PINNDeltaNet(W=W,hidden=128)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} param")

    print(f"\n=== EGİTİM (DS{TRAIN_DS}, delta yaklasimi) ===\n")
    history=train(model,train_data,sc_imu,sc_delta,
                  n_epochs=200,batch=128)

    print(f"\n=== TEST (DS{TEST_DS}) ===")
    pos_pred=predict_cumulative(
        model,t_t,imu_t,head_t,pos_t,delta_t,mask_t,
        sc_imu,sc_delta)
    err,out_idx=evaluate(pos_pred,pos_t,mask_t,
                         label=f"DS{TEST_DS}")

    ate=np.mean(err)
    ate_out=np.mean(err[out_idx]) if len(out_idx) else 0
    plot(pos_pred,pos_t,err,out_idx,t_t,
         history,TEST_DS,ate,ate_out)

    torch.save({'model':model.state_dict(),
                'sc_imu':sc_imu,'sc_delta':sc_delta},
               'pinn_delta_model.pth')
    print("Model: pinn_delta_model.pth")


if __name__=='__main__':
    main()
