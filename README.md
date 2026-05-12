# Kızılötesi Görüntü Anomali Tespiti

Infrared Image Anomaly Detection sistemi - FastAPI backend ve Flutter Web frontend kullanarak geliştirilmiş anomali tespiti uygulaması.

## Proje Yapısı

```
ThesisApp/
├── backend/
│   ├── main.py              # FastAPI ana uygulama
│   ├── models/              # Pydantic modelleri
│   ├── utils/               # Yardımcı fonksiyonlar
│   └── requirements.txt     # Python bağımlılıkları
├── frontend/
│   ├── lib/                 # Flutter kaynak kodları
│   ├── pubspec.yaml         # Flutter bağımlılıkları
│   └── ...
└── README.md
```

## Backend Kurulumu

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Sunucu `http://localhost:8000` adresinde çalışacaktır.

## Frontend Kurulumu

```bash
cd frontend
flutter pub get
flutter run -d web
```

Web uygulaması `http://localhost:54321` adresinde açılacaktır (port değişebilir).

## API Endpoints

- `GET /` - API ana sayfası
- `GET /health` - Sistem durumu kontrolü
