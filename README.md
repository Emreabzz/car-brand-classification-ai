# 🚗 Car Brand Classification AI

**EfficientNet-B3 tabanlı Derin Öğrenme ile Araç Marka Tanıma Sistemi**

Bu proje, araç görsellerinden **marka sınıflandırması** yapmak için
PyTorch kullanılarak geliştirilmiş bir **Derin Öğrenme (Deep Learning)**
uygulamasıdır.\
Eğitilmiş model, **Flask tabanlı bir web arayüzü** üzerinden
kullanıcıdan alınan görseller üzerinde tahmin yapmaktadır.

------------------------------------------------------------------------

## 📌 Proje Özeti

-   🔍 **Problem:** Araç görselinden marka tahmini\
-   🧠 **Model:** EfficientNet-B3 (Transfer Learning + Custom
    Classifier)\
-   ⚙️ **Framework:** PyTorch\
-   🌐 **Web Backend:** Flask\
-   🎨 **Frontend:** HTML, CSS, JavaScript\
-   📊 **Çıktı:** Top-5 marka tahmini + güven oranları

------------------------------------------------------------------------

## 📂 Proje Yapısı

    car-brand-classification-ai/
    │
    ├── app.py
    ├── model.py
    ├── best_model.pth
    ├── class_names.txt
    ├── requirements.txt
    │
    ├── templates/
    │   └── index.html
    │
    └── static/
        ├── style.css
        └── app.js

------------------------------------------------------------------------

## 🧠 Model Mimarisi

-   EfficientNet-B3 backbone
-   ImageNet ön-eğitimli
-   Özel classifier katmanları
-   LayerNorm + Dropout ile regularization

------------------------------------------------------------------------

## 🏷️ Desteklenen Araç Markaları

Toplam **33 sınıf** desteklenmektedir:

Acura, Aston Martin, Audi, Bentley, BMW, Buick, Cadillac, Chevrolet,
Chrysler, Dodge, FIAT, Ford, GMC, Honda, Hyundai, INFINITI, Jaguar,
Jeep, Kia, Land Rover, Lexus, Lincoln, Mazda, Mercedes-Benz, MINI,
Mitsubishi, Nissan, Porsche, Ram, Subaru, Toyota, Volkswagen, Volvo

------------------------------------------------------------------------

## 🚀 Kurulum

``` bash
pip install -r requirements.txt
python app.py
```

Tarayıcıdan aç:

    http://127.0.0.1:5000

------------------------------------------------------------------------

## 🌍 Deploy Mimarisi

-   Frontend: Vercel
-   Backend: Render
-   Model: PyTorch (.pth)

------------------------------------------------------------------------

## 👤 Geliştirici

**Emre Abaz**\
220706022\
Derin Öğrenme Projesi
