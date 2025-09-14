import numpy as np
import os
import json
import random
import config # Menggunakan variabel dari file config.py

class XORDataset:
    """
    Class untuk MEMBACA dataset XOR dari file JSON,
    sesuai dengan path dan pengaturan di config.py.
    """
    def __init__(self):
        """
        Inisialisasi dan langsung muat data dari file.
        """
        self.data = []
        self._load_data()
        
    def _load_data(self):
        """
        Membaca file dataset JSON dan mengubahnya menjadi format
        yang siap digunakan untuk training (numpy array).
        """
        # Mengambil nama file dan direktori dari config.py
        filename = config.DATASET_CONFIG['xor_dataset_file']
        file_path = os.path.join(config.INPUT_DIR, filename)
        
        print(f"Membaca dataset dari: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                dataset = json.load(f)
        except FileNotFoundError:
            print(f"Error: File dataset tidak ditemukan di '{file_path}'")
            print("Pastikan direktori 'data/input/' sudah ada dan berisi file JSON.")
            exit()

        # Ubah data dari JSON menjadi list of tuples (input_array, target_array)
        for sample in dataset['samples']:
            inputs = np.array(sample['input']).reshape(-1, 1)
            targets = np.array(sample['target']).reshape(-1, 1)
            self.data.append((inputs, targets))

    def get_data(self):
        """
        Mengembalikan data training.
        Data akan diacak jika 'shuffle_data' di config adalah True.
        """
        # Buat salinan data agar tidak mengubah urutan asli di self.data
        training_data = self.data.copy()
                    
        return training_data