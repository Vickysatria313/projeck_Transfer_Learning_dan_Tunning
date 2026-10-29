from pathlib import Path
import csv
import math
import time

import torch
import yaml
from ultralytics import YOLO


# =====================================================
# 1. KONFIGURASI
# =====================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data.yaml"

# Jika YOLO26 belum tersedia, otomatis fallback ke YOLO11n
MODEL_PRIMARY = "yolo26n.pt"
MODEL_FALLBACK = "yolo11n.pt"

EPOCHS = 50          # diperpanjang agar perbandingan lebih bermakna
IMAGE_SIZE = 320
BATCH_SIZE = 4
PATIENCE = 30         # diperlonggar agar early-stop tidak asimetris

DEVICE = 0 if torch.cuda.is_available() else "cpu"

PROJECT = ROOT / "runs" / "experiments"

WORKERS = 2
SEED = 42

# Parameter evaluasi eksplisit (default Ultralytics untuk mAP)
VAL_CONF = 0.001
VAL_IOU = 0.6


# =====================================================
# 2. PEMERIKSAAN DATASET & MODEL
# =====================================================

if not DATA.exists():
    raise FileNotFoundError(f"data.yaml tidak ditemukan: {DATA}")

with open(DATA, "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f) or {}

if not data_config.get("test"):
    raise ValueError(
        "Bagian 'test:' belum ada di data.yaml.\n"
        "Tambahkan path gambar test sebelum menjalankan skrip."
    )

PROJECT.mkdir(parents=True, exist_ok=True)


def resolve_model_name():
    """Coba muat model primer; fallback jika gagal."""
    try:
        YOLO(MODEL_PRIMARY)
        return MODEL_PRIMARY
    except Exception as e:
        print(f"PERINGATAN: gagal memuat {MODEL_PRIMARY}: {e}")
        print(f"Fallback ke {MODEL_FALLBACK}")
        YOLO(MODEL_FALLBACK)  # akan raise jika fallback pun gagal
        return MODEL_FALLBACK


MODEL_NAME = resolve_model_name()


def get_class_names():
    names = data_config.get("names")
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return names or []


CLASS_NAMES = get_class_names()


print("=" * 65)
print("YOLO OBJECT DETECTION — BASELINE vs CUSTOM HYPERPARAMETERS")
print("=" * 65)
print(f"Dataset     : {DATA}")
print(f"Model       : {MODEL_NAME}")
print(f"Kelas       : {CLASS_NAMES}")
print(f"Epoch (max) : {EPOCHS}")
print(f"Patience    : {PATIENCE}")
print(f"Image size  : {IMAGE_SIZE}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Device      : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
else:
    print("PERINGATAN: CUDA tidak tersedia. Training menggunakan CPU.")

print("=" * 65)


# =====================================================
# 3. TRAINING
# =====================================================

def train_experiment(experiment_name, custom_hyperparams=False):
    print("\n" + "=" * 65)
    if custom_hyperparams:
        print("EKSPERIMEN 2: PRETRAINED + CUSTOM HYPERPARAMETERS")
    else:
        print("EKSPERIMEN 1: PRETRAINED BASELINE (DEFAULT)")
    print("=" * 65)

    # Muat pretrained bersih dari model yang sama untuk kedua eksperimen
    model = YOLO(MODEL_NAME)

    config = {
        "data": str(DATA),
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        "workers": WORKERS,
        "patience": PATIENCE,
        "seed": SEED,
        "deterministic": True,
        "project": str(PROJECT),
        "name": experiment_name,
        "exist_ok": True,
        "plots": True,
        "save": True,
        "verbose": True,
    }

    if custom_hyperparams:
        # 3 hyperparameter diubah sekaligus: optimizer, LR, intensitas mosaic.
        # Ini "custom config", BUKAN tuning sistematis (grid/random/Bayesian).
        config.update({
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.0005,
            "mosaic": 0.5,
            "close_mosaic": 10,
        })
    else:
        config.update({
            "optimizer": "auto",
        })

    start_time = time.perf_counter()
    model.train(**config)
    training_seconds = time.perf_counter() - start_time
    training_minutes = training_seconds / 60.0

    run_dir = Path(model.trainer.save_dir)
    results_csv = run_dir / "results.csv"
    results_png = run_dir / "results.png"
    best_pt = run_dir / "weights" / "best.pt"

    # Hitung epoch aktual dari results.csv (lebih andal daripada trainer.epoch)
    actual_epochs = None
    if results_csv.exists():
        try:
            with open(results_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                actual_epochs = sum(1 for _ in reader)
        except Exception as e:
            print(f"Peringatan saat menghitung epoch aktual: {e}")
    else:
        print(f"PERINGATAN: results.csv tidak ditemukan: {results_csv}")

    if not best_pt.exists():
        raise FileNotFoundError(f"Bobot best.pt tidak ditemukan: {best_pt}")

    print(f"\nTraining selesai: {experiment_name}")
    print(f"Waktu training : {training_minutes:.2f} menit")
    if actual_epochs is not None:
        print(f"Epoch aktual   : {actual_epochs} / {EPOCHS}")
    print(f"Folder hasil   : {run_dir}")
    print(f"Grafik         : {results_png}")
    print(f"Model terbaik  : {best_pt}")

    return {
        "experiment": experiment_name,
        "run_dir": run_dir,
        "best_pt": best_pt,
        "training_minutes": training_minutes,
        "actual_epochs": actual_epochs,
    }


# =====================================================
# 4. EVALUASI PADA TEST SET
# =====================================================

def evaluate_experiment(experiment):
    name = experiment["experiment"]
    best_pt = experiment["best_pt"]

    print("\n" + "=" * 65)
    print(f"TESTING: {name}")
    print("=" * 65)

    model = YOLO(str(best_pt))

    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        conf=VAL_CONF,
        iou=VAL_IOU,
        plots=True,
        project=str(PROJECT),
        name=f"{name}_test",
        exist_ok=True,
        verbose=True,
    )

    box = metrics.box
    latency_ms = metrics.speed.get("inference", float("nan"))

    row = {
        "Eksperimen": name,
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "Precision (P)": float(box.mp),
        "Recall (R)": float(box.mr),
        "Latency (ms)": float(latency_ms),
        "Waktu Latih (menit)": float(experiment["training_minutes"]),
        "Epoch Aktual": experiment.get("actual_epochs", "-"),
    }

    # Metrik mAP50-95 per kelas
    per_class_rows = []
    try:
        maps = box.maps  # array mAP50-95 per kelas
        class_indices = list(box.ap_class_index)
        for idx, m in zip(class_indices, maps):
            idx_int = int(idx)
            class_name = (
                CLASS_NAMES[idx_int]
                if idx_int < len(CLASS_NAMES)
                else str(idx_int)
            )
            per_class_rows.append({
                "Eksperimen": name,
                "Kelas": class_name,
                "mAP50-95": float(m),
            })
    except Exception as e:
        print(f"Peringatan: metrik per kelas tidak tersedia ({e})")

    print("\nHASIL TEST:")
    for key, value in row.items():
        if isinstance(value, (float, int)):
            print(f"{key:24s}: {value:.6f}")
        else:
            print(f"{key:24s}: {value}")

    if per_class_rows:
        print("\nmAP50-95 per kelas:")
        for r in per_class_rows:
            print(f"  - {r['Kelas']:15s}: {r['mAP50-95']:.4f}")

    print(f"\nGrafik/hasil test: {metrics.save_dir}")

    return row, per_class_rows


# =====================================================
# 5. SIMPAN CSV
# =====================================================

def save_comparison_csv(rows):
    output_csv = PROJECT / "hasil_perbandingan_test.csv"
    columns = [
        "Eksperimen",
        "mAP50",
        "mAP50-95",
        "Precision (P)",
        "Recall (R)",
        "Latency (ms)",
        "Waktu Latih (menit)",
        "Epoch Aktual",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    print(f"\nCSV perbandingan disimpan: {output_csv}")
    return output_csv


def save_per_class_csv(per_class_rows):
    if not per_class_rows:
        return None
    output_csv = PROJECT / "metrik_per_kelas_test.csv"
    columns = ["Eksperimen", "Kelas", "mAP50-95"]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in per_class_rows:
            writer.writerow(row)
    print(f"CSV metrik per kelas disimpan: {output_csv}")
    return output_csv


# =====================================================
# 6. ANALISIS OTOMATIS
# =====================================================

def write_analysis(rows, per_class_rows):
    report_path = PROJECT / "analisis_hasil.txt"

    valid_rows = [r for r in rows if math.isfinite(r["mAP50-95"])]

    lines = [
        "ANALISIS HASIL EKSPERIMEN OBJECT DETECTION",
        "=" * 60,
        f"Model awal  : {MODEL_NAME}",
        f"Epoch (max) : {EPOCHS} (patience {PATIENCE})",
        f"Image size  : {IMAGE_SIZE}",
        f"Batch size  : {BATCH_SIZE}",
        f"Kelas       : {', '.join(map(str, CLASS_NAMES))}",
        "",
        "CATATAN METODOLOGIS:",
        "- Eksperimen 2 adalah KONFIGURASI CUSTOM (3 hyperparameter diubah:",
        "  optimizer, learning rate, dan intensitas mosaic).",
        "- Ini BUKAN hasil hyperparameter tuning sistematis (grid/random/Bayesian).",
        "- Perubahan multi-variabel berarti selisih performa tidak dapat",
        "  diatribusikan ke satu hyperparameter tertentu.",
        "",
        "HASIL PENGUJIAN PADA TEST SET",
        "-" * 60,
    ]

    for row in rows:
        lines.extend([
            f"\n{row['Eksperimen']}",
            f"mAP50              : {row['mAP50']:.4f}",
            f"mAP50-95           : {row['mAP50-95']:.4f}",
            f"Precision          : {row['Precision (P)']:.4f}",
            f"Recall             : {row['Recall (R)']:.4f}",
            f"Latency (ms/img, batch={BATCH_SIZE}) : {row['Latency (ms)']:.4f}",
            f"Waktu latih (menit): {row['Waktu Latih (menit)']:.2f}",
            f"Epoch aktual       : {row.get('Epoch Aktual', '-')} / {EPOCHS}",
        ])

    if per_class_rows:
        lines.extend(["", "mAP50-95 PER KELAS", "-" * 60])
        by_exp = {}
        for r in per_class_rows:
            by_exp.setdefault(r["Eksperimen"], []).append(r)
        for exp, items in by_exp.items():
            lines.append(f"\n{exp}:")
            for r in items:
                lines.append(f"  - {r['Kelas']:20s}: {r['mAP50-95']:.4f}")

    if valid_rows:
        best = max(valid_rows, key=lambda r: r["mAP50-95"])
        lines.extend([
            "",
            "INTERPRETASI",
            "-" * 60,
            (
                f"Berdasarkan mAP50-95, nilai tertinggi diperoleh oleh "
                f"{best['Eksperimen']} ({best['mAP50-95']:.4f})."
            ),
            "",
            "Arti metrik:",
            "- mAP50    : kualitas deteksi pada IoU 0.50.",
            "- mAP50-95 : kualitas deteksi pada IoU 0.50 sampai 0.95.",
            "- Precision: proporsi prediksi positif yang benar.",
            "- Recall   : proporsi objek aktual yang berhasil ditemukan.",
            "- Latency  : waktu inference rata-rata per gambar (batch-based).",
            "- Waktu latih: durasi model.train() berjalan.",
            "",
            "Panduan memilih model:",
            "- mAP50 dan mAP50-95 lebih tinggi umumnya lebih baik.",
            "- Precision lebih tinggi -> lebih sedikit false positive.",
            "- Recall lebih tinggi -> lebih sedikit objek terlewat.",
            "- Latency dan waktu latih lebih rendah -> lebih cepat.",
            "- Untuk deteksi kebakaran, utamakan Recall (api/asap tidak",
            "  boleh terlewat), tetapi tetap evaluasi false alarm.",
            "",
            "CATATAN:",
            "Kesimpulan ini hanya berdasarkan metrik pada test set.",
            "Periksa juga confusion matrix, kurva PR, dan contoh prediksi.",
            "Hasil ini tidak menjamin performa di kondisi nyata.",
        ])

        baseline = next(
            (r for r in rows if "baseline" in r["Eksperimen"].lower()), None
        )
        custom = next(
            (r for r in rows if "custom" in r["Eksperimen"].lower()), None
        )

        if baseline and custom:
            delta_map = (custom["mAP50-95"] - baseline["mAP50-95"]) * 100
            delta_recall = (custom["Recall (R)"] - baseline["Recall (R)"]) * 100
            delta_precision = (custom["Precision (P)"] - baseline["Precision (P)"]) * 100

            lines.extend([
                "",
                "PERBANDINGAN KONFIGURASI CUSTOM TERHADAP BASELINE",
                "-" * 60,
                f"Perubahan mAP50-95 : {delta_map:+.2f} poin persentase",
                f"Perubahan Recall   : {delta_recall:+.2f} poin persentase",
                f"Perubahan Precision: {delta_precision:+.2f} poin persentase",
            ])

            if delta_map > 0:
                lines.append("mAP50-95 meningkat pada konfigurasi custom.")
            elif delta_map < 0:
                lines.append("mAP50-95 menurun pada konfigurasi custom.")
            else:
                lines.append("mAP50-95 tidak berubah pada pembulatan ini.")

            lines.append(
                "Ingat: perbedaan ini akibat perubahan 3 hyperparameter "
                "sekaligus, bukan efek satu hyperparameter."
            )
    else:
        lines.extend([
            "",
            "Metrik mAP50-95 yang valid tidak tersedia.",
            "Periksa log evaluasi, label, dan konfigurasi dataset.",
        ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Analisis disimpan: {report_path}")
    return report_path


# =====================================================
# 7. JALANKAN SEMUA PROSES
# =====================================================

if __name__ == "__main__":

    # Tahap 1: training kedua eksperimen
    experiments = []

    experiments.append(
        train_experiment(
            experiment_name="exp1_pretrained_baseline",
            custom_hyperparams=False,
        )
    )

    experiments.append(
        train_experiment(
            experiment_name="exp2_custom_hyperparams",
            custom_hyperparams=True,
        )
    )

    # Tahap 2: evaluasi kedua best.pt pada test set
    test_rows = []
    per_class_all = []

    for experiment in experiments:
        row, per_class = evaluate_experiment(experiment)
        test_rows.append(row)
        per_class_all.extend(per_class)

    # Tahap 3: simpan CSV & analisis
    save_comparison_csv(test_rows)
    save_per_class_csv(per_class_all)
    write_analysis(test_rows, per_class_all)

    print("\n" + "=" * 65)
    print("SEMUA PROSES SELESAI")
    print(f"Periksa folder: {PROJECT}")
    print("1. results.png di setiap folder training")
    print("2. results.csv untuk metrik per epoch")
    print("3. hasil_perbandingan_test.csv")
    print("4. metrik_per_kelas_test.csv")
    print("5. analisis_hasil.txt")
    print("=" * 65)