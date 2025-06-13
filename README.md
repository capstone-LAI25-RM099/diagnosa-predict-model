# Diagnosa Predict Application

## Deskripsi Proyek

Aplikasi **Diagnosa Predict App** merupakan sebuah sistem prediksi penyakit berbasis gejala yang dibangun menggunakan machine learning. Sistem ini terdiri dari dua bagian utama:

- **Model Machine Learning (Backend API)**: Membangun model klasifikasi penyakit berdasarkan dataset gejala yang telah diproses dan melakukan deployment model sebagai REST API menggunakan Flask.
- **Frontend Interface**: Antarmuka pengguna berbasis web menggunakan ReactJS dan Vite untuk mempermudah pengguna dalam memasukkan gejala serta menerima hasil diagnosa secara cepat.

Proyek ini bertujuan untuk memudahkan masyarakat dalam melakukan *self-diagnosis* awal dengan memasukkan gejala yang dialami, sehingga bisa menjadi pertimbangan awal sebelum konsultasi ke tenaga medis profesional.

---

## Fitur Utama

- Prediksi penyakit berdasarkan gejala.
- Multiple input gejala.
- Model machine learning berbasis Random Forest Classifier.
- Tampilan antarmuka sederhana dan mudah digunakan.
- API service berbasis Flask.
- Dokumentasi dan deployment-ready.

---

# Backend: Diagnosa Predict Model

## Tech Stack

- Python 3.x
- Flask
- Scikit-learn
- Pandas
- Numpy
- Gunicorn
- Heroku Deployment (opsional)
- Dataset symptom-disease

## Struktur Folder

```bash
diagnosa-predict-model/
│
├── dataset/               # Dataset dan file preprocessing
├── model/                 # Model machine learning hasil training
├── __pycache__/           # Cache python
├── notebook.ipynb         # Notebook training model
├── app.py                 # Main Flask App untuk API
├── requirements.txt       # Dependensi python
├── Procfile               # Deployment Heroku
└── README.md              # Dokumentasi proyek
```

## Cara Menjalankan Backend
Clone repository:
```
git clone https://github.com/username/diagnosa-predict-model.git
cd diagnosa-predict-model
```

Install dependency:
```
pip install -r requirements.txt
```

Jalankan aplikasi Flask:
```
uvicorn app:app --reload 
```
Aplikasi akan berjalan di 
`http://127.0.0.1:8000/symptoms`
