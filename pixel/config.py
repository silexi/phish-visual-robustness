# config.py

# --- Ağırlık Ayarları ---
# Bu ağırlıklar, nihai benzerlik skorunu hesaplarken kullanılır. Toplamları 1.0 olmalıdır.
WEIGHTS = {
    "roi_weight": 0.7,       # ROI'nin tam görüntüye göre ağırlığı
    "hash_weight": 0.5,      # Perseptüel hash metriklerinin toplamdaki ağırlığı
    "color_weight": 0.15,    # Renk histogramı metriğinin toplamdaki ağırlığı
    "structure_weight": 0.1, # Yapısal Benzerlik (SSIM) metriğinin ağırlığı
    "lbp_weight": 0.15,      # Yerel İkili Örüntü (LBP) metriğinin ağırlığı
    "hog_weight": 0.1,       # Yönlendirilmiş Gradyanların Histogramı (HOG) metriğinin ağırlığı
}

# --- Hash Ayarları ---
HASH_CONFIG = {
    "dct_scales": [32, 64],
    "wavelet_scales": [32, 64],
    "highfreq_factor": 4,
    "wavelet": 'haar',
    "dct_weights": [0.6, 0.4],
    "wavelet_weights": [0.6, 0.4]
}

# --- Özellik Çıkarma Ayarları ---
FEATURE_CONFIG = {
    "ssim_size": (256, 256),
    "lbp_points": 8,
    "lbp_radius": 1,
    "hog_orientations": 9,
    "hog_pixels_per_cell": (8, 8),
    "hog_cells_per_block": (2, 2)
}

# --- Varsayılan Yollar ---
DEFAULT_PATHS = {
    "reference": "./screenshots/phishing/fake_microsoft.png",
    "target_folder": "./screenshots/legitimate",
    "output_dir": "results"
}