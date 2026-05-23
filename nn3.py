import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.color import rgb2lab, lab2rgb
import gdown

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================

# Локальные пути (файлы уже есть локально)
LOCAL_MODEL_DIR = "model/2026"
CAFFE_PROTO = os.path.join(LOCAL_MODEL_DIR, "colorization_deploy_v2.prototxt")
CAFFE_PTS = os.path.join(LOCAL_MODEL_DIR, "pts_in_hull.npy")

# Пути для файлов с Google Drive
KERAS_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "colorizer_best.keras")
CAFFE_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "colorization_release_v2.caffemodel")

# Прямые ссылки на файлы в Google Drive (ЗАМЕНИТЕ НА ВАШИ ID!)
KERAS_MODEL_URL = "https://drive.google.com/uc?export=download&id=1oHYVnHFVPD5i_WE901aS71QC2ebl_CVQ"
CAFFE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1sWJ1QJvNwlGBOrrROf39rWcp3vJQj3IZ"

# Настройки страницы
st.set_page_config(page_title='AI Colorizer', layout='wide')
st.title('AI Colorizer')
st.write('Загрузите изображение. Результат будет приведён к читаемому размеру с сохранением пропорций.')


# Максимальный размер стороны результата
MAX_DISPLAY_SIZE = 1024

# Фиксированный баланс смешивания
MIX_RATIO = 0.5

# Создаём папку для моделей, если её нет
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ==========================================
# 2. ФУНКЦИЯ ДЛЯ СКАЧИВАНИЯ ФАЙЛОВ
# ==========================================

def download_from_gdrive(url, output_path):
    """Скачивает файл с Google Drive, если его нет локально"""
    if os.path.exists(output_path):
        st.info(f"Файл уже существует локально: {output_path}")
        return True

    try:
        st.info(f"Скачивание {output_path} с Google Drive...")
        gdown.download(url, output_path, quiet=False)
        st.success(f"Успешно скачано: {output_path}")
        return True
    except Exception as e:
        st.error(f"Ошибка при скачивании {output_path}: {e}")
        return False

# ==========================================
# 3. ЗАГРУЗКА МОДЕЛЕЙ
# ==========================================

@st.cache_resource
def loading():
    """Загружает модели — локальные и с Google Drive"""
    models = {}

    # Скачиваем Keras-модель с Google Drive
    if download_from_gdrive(KERAS_MODEL_URL, KERAS_MODEL_PATH):
        try:
            models['keras'] = load_model(KERAS_MODEL_PATH)
            st.success("Keras модель успешно загружена!")
        except Exception as e:
            st.warning(f"Не удалось загрузить Keras модель: {e}")

    # Caffe-модель: скачиваем с Google Drive
    if download_from_gdrive(CAFFE_MODEL_URL, CAFFE_MODEL_PATH):
        # Проверяем наличие остальных файлов Caffe (они уже локальны)
        if os.path.exists(CAFFE_PROTO) and os.path.exists(CAFFE_PTS):
            try:
                net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO, CAFFE_MODEL_PATH)
                pts = np.load(CAFFE_PTS)
                pts = pts.transpose().reshape(2, 313, 1, 1)
                layer1 = net.getLayerId("class8_ab")
                layer2 = net.getLayerId("conv8_313_rh")
                net.getLayer(layer1).blobs = [pts.astype("float32")]
                net.getLayer(layer2).blobs = [np.full([1, 313], 2.606, dtype="float32")]
                models['caffe'] = net
                st.success("Caffe модель успешно загружена!")
            except Exception as e:
                st.warning(f"Не удалось загрузить Caffe модель: {e}")
        else:
            st.error("Не найдены локальные файлы Caffe: .prototxt или .npy")

    return models

models_dict = loading()

if not models_dict:
    st.error("Модели не найдены. Проверьте наличие локальных файлов и корректность ссылок на Google Drive.")
    st.stop()

# ==========================================
# 4. ФУНКЦИИ ОБРАБОТКИ
# ==========================================


def get_keras_ab_small(img_pil):
    """Возвращает AB каналы (256x256) от Keras"""
    IMG_SIZE = 256
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_arr = np.array(img_resized, dtype=float) / 255.0
    lab = rgb2lab(img_arr)
    L_channel = lab[:, :, 0]
    X_input = np.stack((L_channel, L_channel, L_channel), axis=-1)
    X_input = (X_input / 50) - 1
    X_input = X_input.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    SATURATION_MULTIPLIER = 1.4
    pred_ab = models_dict['keras'].predict(X_input, verbose=0)[0] * 128 * SATURATION_MULTIPLIER
    return pred_ab

def get_caffe_ab_small(img_pil):
    """Возвращает AB каналы (256x256) от Caffe"""
    open_cv_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    normalized = open_cv_image.astype("float32") / 255.0
    lab_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab_image, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    models_dict['caffe'].setInput(cv2.dnn.blobFromImage(L))
    ab = models_dict['caffe'].forward()[0, :, :, :].transpose((1, 2, 0))
    ab_256 = cv2.resize(ab, (256, 256))
    return ab_256

def blend_ab_channels(ab1, ab2, ratio=MIX_RATIO):
    """Смешивает два AB канала с заданным коэффициентом"""
    if ab1 is None and ab2 is None:
        return None
    elif ab1 is None:
        return ab2
    elif ab2 is None:
        return ab1
    else:
        return (1 - ratio) * ab1 + ratio * ab2

def reconstruct_color(original_image, ab_channel):
    """Восстанавливает цветное изображение из L и AB каналов"""
    orig_w, orig_h = original_image.size
    img_resized = original_image.resize((256, 256), Image.BILINEAR)
    img_arr = np.array(img_resized, dtype=float) / 255.0
    lab = rgb2lab(img_arr)
    L_small = lab[:, :, 0]

    # Создаём LAB изображение
    colorized_lab = np.zeros((256, 256, 3))
    colorized_lab[:, :, 0] = L_small
    colorized_lab[:, :, 1:] = ab_channel

    # Конвертируем обратно в RGB
    rgb_small = lab2rgb(colorized_lab)

    # Приводим к исходному размеру
    rgb_pil = Image.fromarray((rgb_small * 255).astype(np.uint8))
    rgb_final = rgb_pil.resize((orig_w, orig_h), Image.LANCZOS)

    return rgb_final

# ==========================================
# 5. ИНТЕРФЕЙС И ЛОГИКА ОБРАБОТКИ
# ==========================================

uploaded_file = st.file_uploader(
    'Загрузите изображение',
    type=['jpeg', 'png', 'jpg', 'bmp']
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert('RGB')
    orig_w, orig_h = original_image.size

    # --- РАСЧЁТ ЦЕЛЕВОГО РАЗМЕРА С СОХРАНЕНИЕМ ПРОПОРЦИЙ ---
    if orig_w > orig_h:
        # Если ширина больше — ограничиваем ширину
        new_w = MAX_DISPLAY_SIZE
        new_h = int(orig_h * (MAX_DISPLAY_SIZE / orig_w))
    else:
        # Если высота больше — ограничиваем высоту
        new_h = MAX_DISPLAY_SIZE
        new_w = int(orig_w * (MAX_DISPLAY_SIZE / orig_h))

    target_size = (new_w, new_h)

    # Отображаем оригинальное изображение (уменьшенное для удобства)
    st.subheader("Оригинальное изображение:")
    st.image(original_image.resize(target_size, Image.LANCZOS), use_column_width=True)

    if st.button("🎨 Раскрасить 🎨"):
        with st.spinner('Обработка...'):
            # Получаем цвета в маленьком размере (256x256)
            ab_k_small = None
            if 'keras' in models_dict:
                ab_k_small = get_keras_ab_small(original_image)

            ab_c_small = None
            if 'caffe' in models_dict:
                ab_c_small = get_caffe_ab_small(original_image)

            # Смешиваем результаты двух моделей
            blended_ab = blend_ab_channels(ab_k_small, ab_c_small)

            if blended_ab is not None:
                # Восстанавливаем цветное изображение
                colorized_image = reconstruct_color(original_image, blended_ab)

                # Уменьшаем для отображения
                display_image = colorized_image.resize(target_size, Image.LANCZOS)

                st.subheader("Раскрашенное изображение:")
                st.image(display_image, use_column_width=True)
            else:
                st.error("Не удалось получить цветовые каналы ни от одной модели. Проверьте загрузку моделей.")

