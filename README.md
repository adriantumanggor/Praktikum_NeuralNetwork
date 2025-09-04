# Single Perceptron Neural Network - AND & OR Gates

Implementasi hands-on single perceptron neural network untuk pembelajaran gerbang logika AND dan OR dengan detailed audit logging.

## 📁 Struktur Project

```
perceptron_project/
├── src/
│   ├── __init__.py              # Package initializer
│   ├── perceptron.py            # Class perceptron utama
│   ├── trainer.py               # Training logic dengan CSV logging
│   └── data_loader.py           # Load training data
├── data/
│   ├── and_gate_training.csv    # Training data AND gate
│   ├── or_gate_training.csv     # Training data OR gate
│   └── results/
│       ├── and_training_log.csv # Detailed epoch log AND gate
│       ├── or_training_log.csv  # Detailed epoch log OR gate
│       └── training_summary.csv # Summary hasil training
├── config.py                    # Konfigurasi parameter
├── main.py                      # Entry point utama
├── requirements.txt             # Dependencies
└── README.md                    # Dokumentasi ini
```

## 🚀 Cara Menjalankan

### 1. Clone atau Download Project
```bash
# Buat direktori project
mkdir perceptron_project
cd perceptron_project
```

### 2. Siapkan File
Salin semua file yang telah dibuat ke dalam struktur folder yang sesuai.

### 3. Jalankan Program
```bash
python main.py
```

## 📊 Output Files

### 1. Training Log CSV
File `and_training_log.csv` dan `or_training_log.csv` berisi log detail setiap epoch:

| epoch | sample_idx | x1 | x2 | bias | w1    | w2    | w_bias | weighted_sum | predicted_output | expected_output | error | weight_updated | converged |
|-------|------------|----|----|------|-------|-------|---------|--------------|------------------|-----------------|-------|----------------|-----------|
| 1     | 1          | 0  | 0  | 1    | -0.2  | 0.3   | -0.1    | -0.1         | 0                | 0               | 0     | False          | False     |
| 1     | 2          | 0  | 1  | 1    | -0.2  | 0.3   | -0.1    | 0.2          | 1                | 0               | -1    | True           | False     |

### 2. Training Summary CSV
File `training_summary.csv` berisi ringkasan hasil training:

| gate_type | epochs_to_converge | final_w1 | final_w2 | final_w_bias | final_accuracy | total_weight_updates |
|-----------|-------------------|----------|----------|--------------|----------------|---------------------|
| AND       | 7                 | 0.5      | 0.5      | -0.7         | 1.0            | 8                   |
| OR        | 4                 | 0.3      | 0.3      | -0.1         | 1.0            | 4                   |

## ⚙️ Konfigurasi

Edit file `config.py` untuk mengubah parameter:

```python
LEARNING_RATE = 0.1        # Learning rate (fixed)
MAX_EPOCHS = 1000          # Maximum epochs
WEIGHT_INIT_RANGE = (-0.5, 0.5)  # Range untuk initial weights
```

## 🧠 Spesifikasi Perceptron

- **Activation Function**: Step function (threshold = 0)
- **Learning Rule**: Perceptron learning rule
- **Weight Update**: `w_new = w_old + learning_rate * error * input`
- **Convergence**: Semua training samples diprediksi benar dalam satu epoch

## 📈 Truth Tables

### AND Gate
| x1 | x2 | Output |
|----|----| -------|
| 0  | 0  | 0      |
| 0  | 1  | 0      |
| 1  | 0  | 0      |
| 1  | 1  | 1      |

### OR Gate
| x1 | x2 | Output |
|----|----| -------|
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 1      |

## 💡 Analisis Hasil

Program akan menghasilkan:

1. **Console Output**: Progress training real-time
2. **Detailed Logs**: Setiap step training dalam CSV
3. **Summary Report**: Perbandingan performa AND vs OR gate
4. **Final Model Testing**: Verifikasi akurasi model final

## 🔍 Fitur Audit

Setiap epoch dicatat dengan detail:
- Input values (x1, x2, bias)
- Current weights (w1, w2, w_bias)
- Weighted sum calculation
- Predicted vs expected output
- Error dan weight update status
- Convergence status

Berguna untuk:
- Memahami proses pembelajaran perceptron
- Debugging training issues
- Analisis konvergensi
- Educational purposes

## 📝 Penggunaan untuk Fedora Linux

Project ini kompatibel dengan Fedora Workstation. Pastikan Python 3.x tersedia:

```bash
# Check Python version
python3 --version

# Run program
python3 main.py
```

## 🤝 Contributing

Silakan fork project ini dan buat improvements sesuai kebutuhan pembelajaran Anda!