# comparator.py

from PIL import Image
import numpy as np
import image_processing as ip
import config

class ImageComparator:
    """
    Bir referans görüntüyü yükler, özelliklerini hesaplar ve diğer görüntülerle
    karşılaştırmak için bir arayüz sağlar.
    """
    def __init__(self, reference_path, roi_method="adaptive"):
        """
        Referans görüntüyü yükler ve özelliklerini önceden hesaplar.
        """
        print(f"Referans Görüntü Yükleniyor: {reference_path}")
        self.reference_path = reference_path
        self.roi_method = roi_method
        
        self.ref_img = Image.open(reference_path)
        self.ref_features = self._extract_all_features(self.ref_img)
        print("Referans özellikleri başarıyla çıkarıldı.")

    def _extract_all_features(self, img):
        """
        Verilen bir görüntüden gerekli tüm özellikleri çıkarır.
        """
        features = {}
        # Tam görüntü özellikleri
        features['full_hash'], features['full_weights'] = ip.calculate_multiresolution_hash_from_pil(
            img, **config.HASH_CONFIG)
        features['color_hist'] = ip.calculate_color_histogram(img)
        features['lbp'] = ip.calculate_lbp_features(img)
        features['hog'] = ip.calculate_hog_features(img)

        # ROI özellikleri
        roi_img = ip.get_roi(img, method=self.roi_method)
        features['roi_hash'], features['roi_weights'] = ip.calculate_multiresolution_hash_from_pil(
            roi_img, **config.HASH_CONFIG)
        features['roi_color_hist'] = ip.calculate_color_histogram(roi_img)
        features['roi_lbp'] = ip.calculate_lbp_features(roi_img)
        features['roi_hog'] = ip.calculate_hog_features(roi_img)
        
        return features

    def compare_with(self, target_path):
        """
        Bir hedef görüntüyü, saklanan referans görüntü ile karşılaştırır.
        """
        target_img = Image.open(target_path)
        tgt_features = self._extract_all_features(target_img)

        # 1. Hash Benzerliği
        sim_full_hash = self._weighted_hash_similarity(self.ref_features['full_hash'], self.ref_features['full_weights'], tgt_features['full_hash'])
        sim_roi_hash = self._weighted_hash_similarity(self.ref_features['roi_hash'], self.ref_features['roi_weights'], tgt_features['roi_hash'])
        hash_similarity = (config.WEIGHTS['roi_weight'] * sim_roi_hash) + ((1 - config.WEIGHTS['roi_weight']) * sim_full_hash)
        
        # 2. Renk Benzerliği
        sim_color = ip.calculate_color_similarity(self.ref_features['color_hist'], tgt_features['color_hist'])
        sim_roi_color = ip.calculate_color_similarity(self.ref_features['roi_color_hist'], tgt_features['roi_color_hist'])
        color_similarity = (config.WEIGHTS['roi_weight'] * sim_roi_color) + ((1 - config.WEIGHTS['roi_weight']) * sim_color)

        # 3. Yapısal Benzerlik (SSIM)
        structure_similarity = ip.calculate_structural_similarity(self.ref_img, target_img, size=config.FEATURE_CONFIG['ssim_size'])

        # 4. LBP Benzerliği
        sim_lbp = ip.calculate_lbp_similarity(self.ref_features['lbp'], tgt_features['lbp'])
        sim_roi_lbp = ip.calculate_lbp_similarity(self.ref_features['roi_lbp'], tgt_features['roi_lbp'])
        lbp_similarity = (config.WEIGHTS['roi_weight'] * sim_roi_lbp) + ((1 - config.WEIGHTS['roi_weight']) * sim_lbp)

        # 5. HOG Benzerliği
        sim_hog = ip.calculate_hog_similarity(self.ref_features['hog'], tgt_features['hog'])
        sim_roi_hog = ip.calculate_hog_similarity(self.ref_features['roi_hog'], tgt_features['roi_hog'])
        hog_similarity = (config.WEIGHTS['roi_weight'] * sim_roi_hog) + ((1 - config.WEIGHTS['roi_weight']) * sim_hog)

        # Nihai Ağırlıklı Skor
        final_similarity = (
            config.WEIGHTS['hash_weight'] * hash_similarity +
            config.WEIGHTS['color_weight'] * color_similarity +
            config.WEIGHTS['structure_weight'] * structure_similarity +
            config.WEIGHTS['lbp_weight'] * lbp_similarity +
            config.WEIGHTS['hog_weight'] * hog_similarity
        )
        
        detailed_results = {
            'hash_similarity': hash_similarity,
            'weighted_color_similarity': color_similarity,
            'structure_similarity': structure_similarity,
            'lbp_similarity': lbp_similarity,
            'hog_similarity': hog_similarity
        }
        
        return final_similarity, detailed_results

    def _weighted_hash_similarity(self, hash1, weights1, hash2):
        """İki hash arasındaki ağırlıklı benzerliği hesaplar."""
        mismatches = (hash1 != hash2).astype(float)
        weighted_diff = np.sum(mismatches * weights1)
        total_weight = np.sum(weights1)
        distance = weighted_diff / total_weight
        return 1 - distance