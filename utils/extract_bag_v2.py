"""
MUN-FRL Bell 412 ROS Bag Extraction Script
rosbags >= 0.10 API

Kullanim:
  python extract_bag_v2.py --bag /path/to/bell412_dataset1.bag
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

def extract_bag(bag_path: str):
    bag_path = Path(bag_path)
    out_dir  = bag_path.parent
    stem     = bag_path.stem

    # ROS1 tip deposu
    typestore = get_typestore(Stores.ROS1_NOETIC)

    imu_rows = []
    mag_rows = []
    ppk_rows = []

    print(f"Bag okunuyor: {bag_path}")
    print("Bu islem birkas dakika surebilir...\n")

    with Reader(bag_path) as reader:
        # Mevcut topic'leri listele
        print("Bag icindeki topic'ler:")
        for conn in reader.connections:
            print(f"  {conn.topic}  [{conn.msgtype}]  {conn.msgcount} mesaj")
        print()

        connections = [c for c in reader.connections
                       if c.topic in (
                           '/imu/data',
                           '/imu/data_stamped',
                           '/imu/mag',
                           '/fix_ppk',
                           '/fix',
                       )]

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            topic = conn.topic
            t = timestamp * 1e-9  # ns -> s

            try:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            except Exception:
                continue

            if topic in ('/imu/data', '/imu/data_stamped'):
                imu_rows.append([
                    t,
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ])

            elif topic == '/imu/mag':
                mag_rows.append([
                    t,
                    msg.vector.x,
                    msg.vector.y,
                    msg.vector.z,
                ])

            elif topic in ('/fix_ppk', '/fix'):
                ppk_rows.append([
                    t,
                    msg.latitude,
                    msg.longitude,
                    msg.altitude,
                    msg.status.status,
                ])

    saved = []

    if imu_rows:
        df = pd.DataFrame(imu_rows, columns=[
            'timestamp_s','acc_x','acc_y','acc_z',
            'gyro_x','gyro_y','gyro_z'])
        df['time_s'] = df['timestamp_s'] - df['timestamp_s'].iloc[0]
        out = out_dir / f"{stem}_imu.csv"
        df.to_csv(out, index=False)
        saved.append(out)
        print(f"IMU       : {len(df)} satir | "
              f"{len(df)/df['time_s'].max():.0f} Hz | "
              f"{df['time_s'].max():.1f} s")
        print(f"  acc_z ort: {df['acc_z'].mean():.4f} m/s2")
    else:
        print("UYARI: IMU verisi bulunamadi!")

    if mag_rows:
        df = pd.DataFrame(mag_rows, columns=[
            'timestamp_s','mag_x','mag_y','mag_z'])
        df['time_s'] = df['timestamp_s'] - df['timestamp_s'].iloc[0]
        out = out_dir / f"{stem}_mag.csv"
        df.to_csv(out, index=False)
        saved.append(out)
        print(f"Manyeto   : {len(df)} satir | "
              f"{len(df)/df['time_s'].max():.0f} Hz")
    else:
        print("UYARI: Manyetometre verisi bulunamadi!")

    if ppk_rows:
        df = pd.DataFrame(ppk_rows, columns=[
            'timestamp_s','latitude','longitude','altitude','status'])
        df['time_s'] = df['timestamp_s'] - df['timestamp_s'].iloc[0]
        out = out_dir / f"{stem}_ppk_gnss.csv"
        df.to_csv(out, index=False)
        saved.append(out)
        print(f"PPK GNSS  : {len(df)} satir | "
              f"fix: {(df['status']>=0).mean()*100:.0f}%")
        print(f"  Lat: [{df['latitude'].min():.6f}, "
              f"{df['latitude'].max():.6f}]")
    else:
        print("UYARI: PPK/GNSS verisi bulunamadi!")

    print(f"\n{'='*50}")
    print("Kaydedilen dosyalar:")
    for f in saved:
        mb = Path(f).stat().st_size / 1e6
        print(f"  {f.name}  ({mb:.1f} MB)")
    print("\nBu CSV dosyalarini Claude'a yukleyebilirsiniz.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True,
                        help='Bag dosyasi yolu')
    args = parser.parse_args()
    extract_bag(args.bag)
