# ================================================================
# README.md (as markdown file)
# ================================================================
"""
# MLP Manual Training Project

Implementasi manual Multi-Layer Perceptron dari scratch untuk memahami algoritma pembelajaran neural network secara mendalam.

## 🎯 Tujuan Project

- Memahami forward propagation step-by-step
- Memahami backpropagation dan chain rule
- Melihat bagaimana gradient descent mengupdate weights
- Logging detail setiap kalkulasi untuk analisis mendalam
- Menyelesaikan XOR problem sebagai studi kasus

## 🏗️ Struktur Project

```
mlp_manual_training/
├── README.md
├── requirements.txt
├── config.py                    # Konfigurasi utama
├── main.py                      # Entry point
├── src/                         # Source code
│   ├── network/                 # Neural network components
│   │   ├── mlp.py              # MLP implementation
│   │   └── activations.py      # Activation functions
│   ├── trainer/                 # Training logic
│   │   ├── trainer.py          # Main trainer
│   │   └── logger.py           # Logging functionality
│   ├── data/                    # Data handling
│   │   └── dataset.py          # Dataset management
│   └── utils/                   # Utilities
│       ├── math_utils.py       # Math functions
│       └── file_utils.py       # File operations
├── data/                        # Data directory
│   ├── input/                   # Input datasets
│   └── results/                 # Output logs and models
│       ├── logs/               # Detailed training logs
│       ├── models/             # Saved models
│       └── plots/              # Visualizations
└── analysis/                    # Analysis tools
    ├── visualizer.py           # Training visualization
    └── analyzer.py             # Log analysis
```

## 🚀 Quick Start

1. **Setup Project:**
   ```bash
   python setup.py
   ```

2. **Jalankan Training:**
   ```bash
   python main.py
   ```

3. **Analisis Hasil:**
   ```bash
   python analysis/analyzer.py
   ```

## 📊 Output yang Dihasilkan

### Epoch Summary Logs
- `data/results/logs/epoch_summary.csv`: Summary loss per epoch

### Detailed Calculation Logs
- `data/results/logs/detailed_logs_epoch_X_sample_Y.csv`: Detail kalkulasi per sample
  - Forward pass calculations
  - Loss calculations  
  - Backpropagation errors
  - Weight dan bias updates

### Model Saves
- `data/results/models/trained_model.json`: Final trained model
- `data/results/models/model_epoch_X.json`: Checkpoint models

### Visualizations
- `data/results/plots/training_analysis.png`: Training curve dan analisis

## ⚙️ Konfigurasi

Edit `config.py` untuk mengubah:
- Network architecture (hidden layer size, learning rate)
- Training parameters (epochs, logging frequency)
- File paths dan naming

## 📈 Fitur Logging

### Forward Pass Logging:
- Weighted sum calculation per neuron
- Activation function output
- Weights dan bias yang digunakan

### Backpropagation Logging:
- Error calculation per layer
- Chain rule application
- Sigmoid derivative computation

### Parameter Update Logging:
- Old vs new weight values
- Gradient calculations
- Learning rate application

## 🧠 Konsep yang Dipelajari

1. **Forward Propagation**: Input → Hidden → Output
2. **Loss Calculation**: MSE antara prediction dan target
3. **Backpropagation**: Error propagation dengan chain rule
4. **Gradient Descent**: Weight update berdasarkan gradient

## 🚀 Quick Start

1. **Jalankan Training:**
   ```bash
   python main.py
   ```

2. **Analisis Hasil:**
   ```bash
   python analysis/analyzer.py
   ```

## 📊 Output yang Dihasilkan

- **Epoch Summary**: `data/results/logs/epoch_summary.csv`
- **Detailed Logs**: `data/results/logs/detailed_logs_epoch_X_sample_Y.csv`  
- **Trained Model**: `data/results/models/trained_model.json`
- **Visualizations**: `data/results/plots/training_analysis.png`

## ⚙️ Konfigurasi

Edit `config.py` untuk mengubah network architecture, training parameters, dan file paths.

## 🧠 XOR Problem Dataset

- [0,0] → [0]
- [0,1] → [1] 
- [1,0] → [1]
- [1,1] → [0]

Network Architecture: 2 → 4 → 1 (2 input, 4 hidden, 1 output)

## 📖 Educational Features

✅ **Forward Pass Logging**: Weighted sum, activation output, weights used  
✅ **Backprop Logging**: Error calculation, chain rule, sigmoid derivative  
✅ **Parameter Updates**: Old/new weights, gradients, learning rate application  
✅ **Loss Tracking**: MSE calculation per epoch  
✅ **Visualizations**: Training curves dan convergence analysis


## 📝 XOR Problem

Dataset yang digunakan:
- Input: [0,0] → Output: [0]
- Input: [0,1] → Output: [1] 
- Input: [1,0] → Output: [1]
- Input: [1,1] → Output: [0]

## 🔧 Dependencies

- Python 3.7+
- matplotlib (untuk visualisasi)
- csv, json, os (built-in modules)

## 📖 Educational Value

Project ini memberikan understanding mendalam tentang:
- Bagaimana neural network "belajar" dari data
- Mengapa XOR problem tidak bisa diselesaikan dengan single layer
- Peran activation function dalam non-linearity
- Dampak learning rate terhadap convergence

## 🎯 Next Steps

Setelah memahami dasar-dasar, Anda bisa:
- Experiment dengan different activation functions
- Coba dataset yang lebih complex
- Implement momentum atau adaptive learning rate
- Add regularization techniques

---

**Happy Learning! 🚀**
"""

