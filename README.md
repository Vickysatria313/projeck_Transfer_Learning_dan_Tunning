ANALISIS HASIL EKSPERIMEN OBJECT DETECTION
============================================================
Model awal  : yolo26n.pt
Epoch (max) : 50 (patience 30)
Image size  : 320
Batch size  : 4
Kelas       : FIRE

CATATAN METODOLOGIS:
- Eksperimen 2 adalah KONFIGURASI CUSTOM (3 hyperparameter diubah:
  optimizer, learning rate, dan intensitas mosaic).
- Ini BUKAN hasil hyperparameter tuning sistematis (grid/random/Bayesian).
- Perubahan multi-variabel berarti selisih performa tidak dapat
  diatribusikan ke satu hyperparameter tertentu.
HASIL PENGUJIAN PADA TEST SET
------------------------------------------------------------

exp1_pretrained_baseline
mAP50              : 0.8737
mAP50-95           : 0.4794
Precision          : 0.8764
Recall             : 0.8439
Latency (ms/img, batch=4) : 6.7215
Waktu latih (menit): 78.25
Epoch aktual       : 50 / 50

exp2_custom_hyperparams
mAP50              : 0.9005
mAP50-95           : 0.4850
Precision          : 0.8565
Recall             : 0.8175
Latency (ms/img, batch=4) : 6.8908
Waktu latih (menit): 77.65
Epoch aktual       : 50 / 50

mAP50-95 PER KELAS
------------------------------------------------------------

exp1_pretrained_baseline:
  - FIRE                : 0.4794

exp2_custom_hyperparams:
  - FIRE                : 0.4850
INTERPRETASI
------------------------------------------------------------
Berdasarkan mAP50-95, nilai tertinggi diperoleh oleh exp2_custom_hyperparams (0.4850).

Arti metrik:
- mAP50    : kualitas deteksi pada IoU 0.50.
- mAP50-95 : kualitas deteksi pada IoU 0.50 sampai 0.95.
- Precision: proporsi prediksi positif yang benar.
- Recall   : proporsi objek aktual yang berhasil ditemukan.
- Latency  : waktu inference rata-rata per gambar (batch-based).
- Waktu latih: durasi model.train() berjalan.

Panduan memilih model:
- mAP50 dan mAP50-95 lebih tinggi umumnya lebih baik.
- Precision lebih tinggi -> lebih sedikit false positive.
- Recall lebih tinggi -> lebih sedikit objek terlewat.
- Latency dan waktu latih lebih rendah -> lebih cepat.
- Untuk deteksi kebakaran, utamakan Recall (api/asap tidak
  boleh terlewat), tetapi tetap evaluasi false alarm.
  
  PERBANDINGAN KONFIGURASI CUSTOM TERHADAP BASELINE
------------------------------------------------------------
Perubahan mAP50-95 : +0.55 poin persentase
Perubahan Recall   : -2.65 poin persentase
Perubahan Precision: -1.99 poin persentase
mAP50-95 meningkat pada konfigurasi custom.
