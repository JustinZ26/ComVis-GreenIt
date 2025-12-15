import gradio as gr
import numpy as np
import joblib
import json
import cv2
import os
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.color import rgb2hsv

# Load model & preprocessors
MODEL_DIR = "model"

svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca_transformer.joblib"))

with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
    class_label = json.load(f)

deskripsi_sampah = {
  "E-waste": {
    "deskripsi": "Sampah elektronik berbahaya. Jangan buang di tempat sampah biasa. Bawa ke drop box e-waste.",
    "tong_warna": "Merah (B3)",
    "dapat_didaur_ulang": "Ya, melalui fasilitas khusus B3",
    "dampak_jika_tidak_diolah": "Pelepasan zat beracun (merkuri, timbal, kadmium) ke tanah dan air. Zat ini mencemari rantai makanan dan sangat berbahaya bagi kesehatan manusia. (Referensi: UNEP/Basel Convention)"
  },
  "Glass": {
    "deskripsi": "Kaca bisa didaur ulang tanpa batas. Pastikan tidak pecah saat dibuang agar aman bagi petugas.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Tidak terurai (inert) dan memenuhi TPA. Pecahan kaca dapat melukai hewan dan petugas, serta berpotensi menyebabkan kebakaran karena efek lensa. (Referensi: Ilmu Lingkungan Material)"
  },
  "Organic Waste": {
    "deskripsi": "Sampah organik (sisa makanan/daun). Bagus untuk dijadikan kompos.",
    "tong_warna": "Hijau (Organik)",
    "dapat_didaur_ulang": "Ya (Diolah menjadi kompos)",
    "dampak_jika_tidak_diolah": "Dalam TPA, penguraian anaerobik menghasilkan gas **metana ($CH_4$)**, yaitu gas rumah kaca yang 25 kali lebih kuat dari karbon dioksida ($CO_2$) dalam memerangkap panas. (Referensi: IPCC/Lembaga Penelitian Lingkungan)"
  },
  "Textiles": {
    "deskripsi": "Limbah tekstil seperti baju bekas. Bisa disumbangkan atau didaur ulang menjadi kain lap.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya (Didaur ulang/Digunakan kembali)",
    "dampak_jika_tidak_diolah": "Membutuhkan lahan TPA yang besar. Tekstil modern melepaskan **serat mikroplastik** saat terurai di lingkungan dan membutuhkan waktu puluhan hingga ratusan tahun. (Referensi: Studi Limbah Tekstil/Microplastic Research)"
  },
  "cardboard": {
    "deskripsi": "Kardus/Karton. Lipat hingga pipih sebelum dibuang untuk menghemat ruang. Bisa didaur ulang menjadi kertas.",
    "tong_warna": "Biru (Kertas)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Memenuhi TPA dan penguraiannya di TPA juga dapat menghasilkan metana jika basah. Daur ulang kardus menghemat energi dan mengurangi penebangan pohon. (Referensi: WWF/Pusat Daur Ulang Kertas)"
  },
  "metal": {
    "deskripsi": "Logam/Kaleng. Cuci bersih sisa makanan sebelum dibuang ke tempat daur ulang.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Logam membutuhkan waktu ratusan tahun untuk terurai. Logam yang berkarat dapat mencemari air tanah dan memerlukan ekstraksi sumber daya alam (penambangan) yang intensif energi. (Referensi: US Geological Survey/Ilmu Material)"
  },
  "paper": {
    "deskripsi": "Kertas. Pastikan kering dan tidak berminyak agar bisa didaur ulang.",
    "tong_warna": "Biru (Kertas)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Menyumbang volume besar di TPA. Kegagalan mendaur ulang berarti peningkatan permintaan kayu dan energi untuk memproduksi kertas baru. (Referensi: Studi Konservasi Energi dan Sumber Daya Alam)"
  },
  "plastic": {
    "deskripsi": "Plastik butuh waktu lama terurai. Pisahkan botol dan gelas plastik untuk didaur ulang.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Membutuhkan ratusan hingga ribuan tahun untuk terurai, mencemari lautan, dan terpecah menjadi **mikroplastik** yang masuk ke rantai makanan dan ekosistem. (Referensi: Jurnal Ilmu Kelautan/Plastics Pollution Coalition)"
  },
  "shoes": {
    "deskripsi": "Sepatu bekas. Jika masih layak pakai, sebaiknya didonasikan.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya (Digunakan kembali/Daur ulang terbatas)",
    "dampak_jika_tidak_diolah": "Terbuat dari material campuran kompleks (karet, kulit, plastik, busa) yang hampir mustahil terurai secara alami, sehingga menumpuk di TPA. (Referensi: Analisis Material Limbah Kompleks)"
  },
  "trash": {
    "deskripsi": "Sampah residu atau lainnya yang sulit didaur ulang. Buang ke tempat sampah umum.",
    "tong_warna": "Abu-abu (Residu)",
    "dapat_didaur_ulang": "Tidak",
    "dampak_jika_tidak_diolah": "Menyebabkan penumpukan di TPA, memerlukan lahan yang terus bertambah, dan menjadi sumber bau tidak sedap, serta lindi (air sampah) yang mencemari lingkungan. (Referensi: Pedoman Pengelolaan TPA)"
  }
}

def extract_traditional_features_from_pil(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (150, 150))

    # HSV Histogram
    hsv = rgb2hsv(img)
    h_hist = np.histogram(hsv[:, :, 0], bins=16, range=(0, 1))[0]
    s_hist = np.histogram(hsv[:, :, 1], bins=16, range=(0, 1))[0]
    v_hist = np.histogram(hsv[:, :, 2], bins=16, range=(0, 1))[0]

    h_hist = h_hist / np.sum(h_hist)
    s_hist = s_hist / np.sum(s_hist)
    v_hist = v_hist / np.sum(v_hist)

    color_features = np.concatenate([h_hist, s_hist, v_hist])

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # GLCM
    bins = np.linspace(0, 256, 9)
    quantized = np.digitize(gray, bins[:-1]) - 1

    glcm = graycomatrix(
        quantized,
        distances=[3, 5],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=8,
        symmetric=True,
        normed=True
    )

    texture_features = np.array([
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'correlation').mean()
    ])

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / gray.size
    edge_features = np.array([edge_density])

    # HOG
    hog_features = hog(
        gray,
        orientations=8,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )

    # LBP
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(
        lbp,
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )
    lbp_hist = lbp_hist / np.sum(lbp_hist)

    feature_vector = np.concatenate([
        color_features,
        texture_features,
        edge_features,
        hog_features,
        lbp_hist
    ])

    return feature_vector.reshape(1, -1)


def predict_input(img):
    features = extract_traditional_features_from_pil(img)

    # Scale + PCA
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # Predict
    pred_idx = svm_model.predict(features_pca)[0]
    pred_label = class_label[int(pred_idx)]

    # Confidence
    decision_scores = svm_model.decision_function(features_pca)
    scores = np.exp(decision_scores) / np.sum(np.exp(decision_scores))

    confidence_dict = {
        class_label[i]: float(scores[0][i])
        for i in range(len(class_label))
    }

    info = deskripsi_sampah.get(pred_label)

    if info:
        markdown = f"""
### Hasil Deteksi: **{pred_label}**

* **📄 Deskripsi:** {info['deskripsi']}
* **🗑️ Tong Sampah:** {info['tong_warna']}
* **♻️ Daur Ulang:** {info['dapat_didaur_ulang']}

---
#### ⚠️ Dampak Jika Tidak Diolah
{info['dampak_jika_tidak_diolah']}
"""
    else:
        markdown = f"### {pred_label}\nInformasi belum tersedia."

    return confidence_dict, markdown


# Gradio UI

demo = gr.Interface(
    fn=predict_input,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Prediksi Kategori"),
        gr.Markdown(label="Saran Pengolahan")
    ],
    title="Klasifikasi Sampah Berbasis SVM",
    description="Model SVM dengan ekstraksi fitur tradisional"
)

demo.launch()