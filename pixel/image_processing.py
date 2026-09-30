# image_processing.py


import numpy as np
import scipy.fftpack
import os
from PIL import Image, ImageOps, ImageFilter
import pywt  # Wavelet dönüşümü için PyWavelets kütüphanesi
import cv2   # Renk histogramı ve yapısal benzerlik için OpenCV
from skimage.metrics import structural_similarity as ssim
import itertools  # Ağırlık kombinasyonları için
from skimage.feature import local_binary_pattern, hog  # LBP ve HOG için
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pandas as pd

def calculate_dct_hash_from_pil(img, hash_size=32, highfreq_factor=4):
    """
    Verilen PIL image üzerinden, ön işleme (grayscale, equalize, blur, resize) yaparak DCT tabanlı hash hesaplar.
    """
    # Ön işleme: grayscale, histogram eşitleme, hafif Gaussian blur
    image = img.convert('L')
    image = ImageOps.equalize(image)
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    # DCT için yeniden boyutlandırma: (hash_size * highfreq_factor) x (hash_size * highfreq_factor)
    img_size = hash_size * highfreq_factor
    image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    
    # 2D DCT uygulaması
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0, norm='ortho'),
                             axis=1, norm='ortho')
    # İlk hash_size x hash_size düşük frekans bileşeni
    dct_low_freq = dct[:hash_size, :hash_size]
    median_val = np.median(dct_low_freq)
    # İsteğe bağlı ağırlıklandırma: örneğin, sol üst köşeye ekstra ağırlık verelim.
    weights_local = np.ones_like(dct_low_freq, dtype=float)
    weights_local[0:2, 0:2] *= 1.5
    dct_hash = (dct_low_freq * weights_local) > median_val
    return dct_hash.astype(int)

def calculate_wavelet_hash_from_pil(img, hash_size=32, wavelet='haar'):
    """
    Verilen PIL image üzerinden, ön işleme (grayscale, equalize, blur, resize)
    yaparak Wavelet tabanlı hash hesaplar.
    """
    # Ön işleme: grayscale, histogram eşitleme, blur
    image = img.convert('L')
    image = ImageOps.equalize(image)
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    # Wavelet için boyutlandırma: DWT sonrası LL bileşeni boyutu hash_size x hash_size olması için
    img_size = hash_size * 2
    image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    
    # 2D DWT: sadece LL (yaklaşım) bileşeni üzerinden hash hesaplama
    coeffs2 = pywt.dwt2(pixels, wavelet)
    LL, (LH, HL, HH) = coeffs2
    median_val = np.median(LL)
    wavelet_hash = LL > median_val
    return wavelet_hash.astype(int)

def calculate_multiresolution_hash_from_pil(img, 
                                            dct_scales=[32, 64], 
                                            highfreq_factor=4, 
                                            wavelet_scales=[32, 64], 
                                            wavelet='haar',
                                            dct_weights=[0.6, 0.4],
                                            wavelet_weights=[0.6, 0.4]):
    """
    Verilen PIL image için, farklı ölçeklerde DCT ve Wavelet hash'lerini hesaplar,
    her ölçeğe ait hash'lere ağırlık atar ve birleştirir.
    
    Returns:
        tuple: (combined_hash, combined_weights)
    """
    combined_hash_parts = []
    combined_weight_parts = []
    
    # DCT hash'leri
    for i, scale in enumerate(dct_scales):
        weight_value = dct_weights[i] if i < len(dct_weights) else 1.0
        dct_hash = calculate_dct_hash_from_pil(img, hash_size=scale, highfreq_factor=highfreq_factor)
        flat_hash = dct_hash.flatten()
        weight_array = np.full(flat_hash.shape, weight_value, dtype=float)
        combined_hash_parts.append(flat_hash)
        combined_weight_parts.append(weight_array)
    
    # Wavelet hash'leri
    for i, scale in enumerate(wavelet_scales):
        weight_value = wavelet_weights[i] if i < len(wavelet_weights) else 1.0
        wavelet_hash = calculate_wavelet_hash_from_pil(img, hash_size=scale, wavelet=wavelet)
        flat_hash = wavelet_hash.flatten()
        weight_array = np.full(flat_hash.shape, weight_value, dtype=float)
        combined_hash_parts.append(flat_hash)
        combined_weight_parts.append(weight_array)
    
    combined_hash = np.concatenate(combined_hash_parts)
    combined_weights = np.concatenate(combined_weight_parts)
    return combined_hash, combined_weights

# --- Renk Histogramı Hesaplama ---
def calculate_color_histogram(img):
    """
    Verilen PIL görüntüsünün renk histogramını hesaplar.
    
    Returns:
        np.ndarray: Normalize edilmiş renk histogramı
    """
    # PIL görüntüsünü OpenCV formatına dönüştür
    img_cv = np.array(img.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy()  # RGB -> BGR
    
    # Renk histogramı hesapla (HSV renk uzayında)
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    
    # Histogramı normalize et
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

# --- Yapısal Benzerlik Hesaplama ---
def calculate_structural_similarity(img1, img2, size=(256, 256)):
    """
    İki PIL görüntüsü arasındaki yapısal benzerliği hesaplar.
    
    Returns:
        float: Yapısal benzerlik skoru (0-1 arası)
    """
    # Görüntüleri aynı boyuta getir
    img1_resized = img1.resize(size, Image.Resampling.LANCZOS).convert('L')
    img2_resized = img2.resize(size, Image.Resampling.LANCZOS).convert('L')
    
    # PIL -> NumPy dizisi
    img1_array = np.array(img1_resized)
    img2_array = np.array(img2_resized)
    
    # Yapısal benzerlik hesapla
    similarity = ssim(img1_array, img2_array, data_range=255)
    return similarity

# --- Renk Benzerliği Hesaplama ---
def calculate_color_similarity(hist1, hist2):
    """
    İki renk histogramı arasındaki benzerliği hesaplar.
    
    Returns:
        float: Benzerlik skoru (0-1 arası)
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# --- Akıllı ROI Belirleme Fonksiyonu ---
def get_roi(img, method="fixed"):
    """
    Verilen PIL image üzerinden, login formunun bulunduğu bölgeyi (ROI) kırpar.
    
    Args:
        img: PIL Image
        method: ROI belirleme yöntemi ("fixed", "adaptive", "form_detection", "content_aware")
    
    Returns:
        PIL Image: Kırpılmış ROI bölgesi
    """
    width, height = img.size
    
    if method == "fixed":
        # Sabit koordinatlar: sol %25, üst %30; sağ %75, alt %80
        left = int(width * 0.25)
        upper = int(height * 0.30)
        right = int(width * 0.75)
        lower = int(height * 0.80)
    
    elif method == "adaptive":
        # Görüntünün içeriğine göre dinamik ROI belirleme
        # Örnek: Görüntünün orta kısmını daha fazla kapsayacak şekilde
        left = int(width * 0.2)
        upper = int(height * 0.2)
        right = int(width * 0.8)
        lower = int(height * 0.8)
    
    elif method == "form_detection":
        # Form elemanlarını tespit etmeye çalışan daha gelişmiş bir yöntem
        # Bu örnekte basit bir yaklaşım kullanıyoruz
        # Gerçek uygulamada makine öğrenimi veya daha gelişmiş görüntü işleme kullanılabilir
        img_cv = np.array(img.convert('RGB'))
        img_cv = img_cv[:, :, ::-1].copy()  # RGB -> BGR
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Kenar tespiti
        edges = cv2.Canny(gray, 50, 150)
        
        # Konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # En büyük konturları al (potansiyel form alanları)
        if contours:
            # Konturları alanlarına göre sırala
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # İlk birkaç büyük konturu al
            large_contours = contours[:min(5, len(contours))]
            
            # Konturların sınırlayıcı kutularını birleştir
            x_min, y_min = width, height
            x_max, y_max = 0, 0
            
            for contour in large_contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Sınırları kontrol et ve ayarla
            left = max(0, x_min - 10)
            upper = max(0, y_min - 10)
            right = min(width, x_max + 10)
            lower = min(height, y_max + 10)
            
            # Çok küçük veya çok büyük ROI'leri önle
            if (right - left) < width * 0.2 or (lower - upper) < height * 0.2:
                # Varsayılan değerlere dön
                left = int(width * 0.25)
                upper = int(height * 0.30)
                right = int(width * 0.75)
                lower = int(height * 0.80)
        else:
            # Kontur bulunamazsa varsayılan değerleri kullan
            left = int(width * 0.25)
            upper = int(height * 0.30)
            right = int(width * 0.75)
            lower = int(height * 0.80)
    
    elif method == "content_aware":
        # İçerik tabanlı ROI belirleme - görüntüdeki önemli bölgeleri tespit eder
        img_cv = np.array(img.convert('RGB'))
        img_cv = img_cv[:, :, ::-1].copy()  # RGB -> BGR
        
        # 1. Gri tonlama ve bulanıklaştırma
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Kenar tespiti
        edges = cv2.Canny(blurred, 50, 150)
        
        # 3. Morfolojik işlemler - kenarları güçlendir
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 4. Konturları bul
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Konturları filtrele - çok küçük olanları çıkar
        min_contour_area = (width * height) * 0.001  # Görüntü alanının %0.1'i
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        if filtered_contours:
            # 6. Konturların yoğunluk haritasını oluştur
            density_map = np.zeros((height, width), dtype=np.float32)
            
            for contour in filtered_contours:
                # Kontur alanını doldur
                cv2.drawContours(density_map, [contour], 0, 1, -1)
            
            # 7. Yoğunluk haritasını bulanıklaştır
            density_map = cv2.GaussianBlur(density_map, (width//10*2+1, height//10*2+1), 0)
            
            # 8. En yoğun bölgeyi bul
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(density_map)
            
            # 9. En yoğun bölge etrafında ROI oluştur
            center_x, center_y = max_loc
            roi_width = width // 2
            roi_height = height // 2
            
            left = max(0, center_x - roi_width // 2)
            upper = max(0, center_y - roi_height // 2)
            right = min(width, center_x + roi_width // 2)
            lower = min(height, center_y + roi_height // 2)
            
            # 10. ROI'yi genişlet - en az görüntünün %30'unu kapsamalı
            if (right - left) < width * 0.3:
                expand_x = (width * 0.3 - (right - left)) / 2
                left = max(0, int(left - expand_x))
                right = min(width, int(right + expand_x))
            
            if (lower - upper) < height * 0.3:
                expand_y = (height * 0.3 - (lower - upper)) / 2
                upper = max(0, int(upper - expand_y))
                lower = min(height, int(lower + expand_y))
        else:
            # Kontur bulunamazsa varsayılan değerleri kullan
            left = int(width * 0.25)
            upper = int(height * 0.30)
            right = int(width * 0.75)
            lower = int(height * 0.80)
    
    else:
        # Varsayılan olarak sabit koordinatları kullan
        left = int(width * 0.25)
        upper = int(height * 0.30)
        right = int(width * 0.75)
        lower = int(height * 0.80)
    
    return img.crop((left, upper, right, lower))

# --- Gelişmiş Özellik Çıkarma Fonksiyonları ---
def calculate_lbp_features(img, num_points=8, radius=1, method='uniform'):
    """
    Verilen PIL görüntüsünden Yerel İkili Örüntü (LBP) özelliklerini çıkarır.
    
    Args:
        img: PIL Image
        num_points: LBP hesaplaması için komşu nokta sayısı
        radius: LBP hesaplaması için yarıçap
        method: LBP yöntemi ('default', 'ror', 'uniform', 'nri_uniform', 'var')
        
    Returns:
        np.ndarray: LBP histogramı
    """
    # PIL -> NumPy dizisi
    img_cv = np.array(img.convert('L'))
    
    # Görüntüyü yeniden boyutlandır (hızlı hesaplama için)
    img_resized = cv2.resize(img_cv, (128, 128))
    
    # LBP hesapla
    lbp = local_binary_pattern(img_resized, num_points, radius, method)
    
    # Histogram hesapla
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

def calculate_hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Verilen PIL görüntüsünden Histogram of Oriented Gradients (HOG) özelliklerini çıkarır.
    
    Args:
        img: PIL Image
        orientations: Yönelim sayısı
        pixels_per_cell: Hücre başına piksel sayısı
        cells_per_block: Blok başına hücre sayısı
        
    Returns:
        np.ndarray: HOG özellikleri
    """
    # PIL -> NumPy dizisi
    img_cv = np.array(img.convert('L'))
    
    # Görüntüyü yeniden boyutlandır (hızlı hesaplama için)
    img_resized = cv2.resize(img_cv, (128, 128))
    
    # HOG hesapla
    features = hog(img_resized, 
                  orientations=orientations, 
                  pixels_per_cell=pixels_per_cell,
                  cells_per_block=cells_per_block, 
                  block_norm='L2-Hys',
                  visualize=False)
    
    return features

def calculate_lbp_similarity(hist1, hist2):
    """
    İki LBP histogramı arasındaki benzerliği hesaplar.
    
    Returns:
        float: Benzerlik skoru (0-1 arası)
    """
    # Histogram kesişimi
    intersection = np.sum(np.minimum(hist1, hist2))
    return intersection

def calculate_hog_similarity(feat1, feat2):
    """
    İki HOG özellik vektörü arasındaki benzerliği hesaplar.
    
    Returns:
        float: Benzerlik skoru (0-1 arası)
    """
    # Kosinüs benzerliği
    dot_product = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    similarity = dot_product / (norm1 * norm2)
    return (similarity + 1) / 2  # [-1, 1] -> [0, 1]


# Diğer tüm `calculate_*` fonksiyonları bu dosyada yer almalıdır.




def calculate_f1_score(results, threshold=0.7):
    """
    Sonuçlar üzerinden F1 skorunu hesaplar.
    
    Args:
        results: compare_images fonksiyonunun döndürdüğü sonuçlar
        threshold: Benzerlik eşiği (varsayılan: 0.7)
    
    Returns:
        dict: F1 skoru ve diğer metrikler
    """
    # Gerçek pozitif, yanlış pozitif, gerçek negatif, yanlış negatif sayılarını hesapla
    tp = 0  # True Positive: Benzerlik >= threshold ve phishing
    fp = 0  # False Positive: Benzerlik >= threshold ama legitimate
    tn = 0  # True Negative: Benzerlik < threshold ve legitimate
    fn = 0  # False Negative: Benzerlik < threshold ama phishing
    
    for filename, similarity, _ in results:
        # Dosya adından phishing/legitimate durumunu belirle
        is_phishing = "phishing" in filename.lower()
        
        if similarity >= threshold:
            if is_phishing:
                tp += 1
            else:
                fp += 1
        else:
            if is_phishing:
                fn += 1
            else:
                tn += 1
    
    # Hassasiyet (Precision)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Duyarlılık (Recall)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Skoru
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Doğruluk (Accuracy)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def calculate_matching_score(results, reference_name):
    """
    Sonuçlar üzerinden doğru eşleştirme oranını hesaplar.
    Örneğin: fake_microsoft.png'in microsoft.png, outlook.png veya office.png ile en yüksek benzerliği göstermesi beklenir.
    
    Args:
        results: compare_images fonksiyonunun döndürdüğü sonuçlar
    
    Returns:
        dict: Eşleştirme metrikleri
    """
    if not results:
        return {
            'matching_score': 0,
            'is_correct_match': False,
            'top_match': None,
            'top_similarity': 0,
            'target_site': 'unknown',
            'related_services': [],
            'matched_service': None
        }
    
    # Referans görüntünün adından hedef siteyi belirle referans görüntü örnek fake_microsoft.png
    target_site = None
    related_services = []
    reference_name = reference_name.split(".")[0]
    reference_name = reference_name.split("_")[1]

    print("reference_name:", reference_name)
    print("target_site:", target_site)
    print("related_services:", related_services)
    
    # Referans görüntüden hedef siteyi ve ilgili servisleri çıkar
    if "microsoft" in reference_name:
        target_site = "microsoft"
        related_services = ["outlook", "office", "azure", "onedrive", "teams", "sharepoint", "windows"]
    elif "paypal" in reference_name:
        target_site = "paypal"
        related_services = ["paypal", "venmo", "xoom"]
    elif "gmail" in reference_name:
        target_site = "gmail"
        related_services = ["gmail", "google", "g-suite", "workspace"]
    elif "dropbox" in reference_name:
        target_site = "dropbox"
        related_services = ["dropbox", "paper", "hello sign"]
    # Diğer siteler için benzer kontroller eklenebilir

    if target_site is None:
        return {
            'matching_score': 0,
            'is_correct_match': False,
            'top_match': results[0][0] if results else None,
            'top_similarity': results[0][1] if results else 0,
            'target_site': 'unknown',
            'related_services': [],
            'matched_service': None
        }
    
    # En yüksek benzerliğe sahip görüntüyü bul
    top_match = results[0]
    top_filename = top_match[0].lower()
    
    # Doğru eşleşme kontrolü - hedef site veya ilgili servislerden herhangi biriyle eşleşme
    is_correct_match = any(service in top_filename for service in [target_site] + related_services)
    
    # Hangi servisle eşleştiğini bul
    matched_service = None
    for service in [target_site] + related_services:
        if service in top_filename:
            matched_service = service
            break
    
    return {
        'matching_score': 1.0 if is_correct_match else 0.0,
        'is_correct_match': is_correct_match,
        'top_match': top_match[0],
        'top_similarity': top_match[1],
        'target_site': target_site,
        'related_services': related_services,
        'matched_service': matched_service
    }
