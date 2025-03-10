import pandas as pd
import torch
import torchaudio
import yt_dlp
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['link', 'category'])
    categories = df['category'].unique()
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    df['label'] = df['category'].map(category_to_idx)
    return df, category_to_idx


def download_and_load_audio(url, output_dir="temp_audio", ffmpeg_path="ffmpeg/bin/ffmpeg.exe", max_length=80000):
    try:
        Path(output_dir).mkdir(exist_ok=True)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_dir}/audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'ffmpeg_location': ffmpeg_path
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Скачиваем аудио с {url}")
            ydl.download([url])
        audio_file = next(Path(output_dir).glob("audio*.wav"))
        waveform, sample_rate = torchaudio.load(str(audio_file))

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)
            sample_rate = 16000

        # Если стерео, преобразуем в моно, усредняя каналы
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Обрезаем или дополняем waveform до фиксированной длины
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]  # Обрезка
        elif waveform.shape[1] < max_length:
            padding = torch.zeros(waveform.shape[0], max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)  # Дополнение

        os.remove(audio_file)
        return waveform, sample_rate
    except Exception as e:
        print(f"Ошибка при обработке {url}: {str(e)}")
        return None, None


class YouTubeAudioDataset(Dataset):
    def __init__(self, links, labels, processor, max_length=80000):
        self.links = links
        self.labels = labels
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        link = self.links[idx]
        label = self.labels[idx]
        waveform, sample_rate = download_and_load_audio(link, max_length=self.max_length)
        if waveform is None:
            # Возвращаем пустые данные в случае ошибки (обратите внимание на attention_mask)
            return (
                torch.zeros(1, self.max_length),
                torch.zeros(1, self.max_length),  # attention_mask из нулей означает, что данные игнорируются
                torch.tensor(label, dtype=torch.long)
            )

        # Преобразуем waveform в numpy массив и убираем лишние размерности
        waveform_np = waveform.squeeze().numpy()  # Преобразуем в 1D numpy массив

        # Обрабатываем waveform с помощью processor
        processed = self.processor(
            waveform_np,  # Передаем 1D numpy массив
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding='max_length',  # Исправлено: padding='max_length' для гарантии attention_mask
            truncation=True,  # Обрезаем, если длиннее max_length
            max_length=self.max_length  # Устанавливаем max_length для обрезки и дополнения
        )

        # Проверяем, что attention_mask присутствует
        if 'attention_mask' not in processed:
            print(f"Предупреждение: attention_mask отсутствует для ссылки {link}. Используем маску из единиц.")
            attention_mask = torch.ones_like(processed.input_values)  # Заполняем единицами как запасной вариант
        else:
            attention_mask = processed.attention_mask

        return (
            processed.input_values.squeeze(0),  # Убираем размерность батча
            attention_mask.squeeze(0),  # Убираем размерность батча
            torch.tensor(label, dtype=torch.long)
        )


def collate_fn(batch):
    # Фильтруем пустые данные
    filtered_batch = [item for item in batch if not torch.all(item[0] == 0)]
    if not filtered_batch:
        # Если все элементы пустые, возвращаем пустой батч
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    input_values = torch.stack([item[0] for item in filtered_batch])
    attention_masks = torch.stack([item[1] for item in filtered_batch])
    labels = torch.stack([item[2] for item in filtered_batch])

    return input_values, attention_masks, labels


def train_and_evaluate(train_loader, test_loader, model, num_epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (input_values, attention_mask, labels) in enumerate(train_loader):
            if input_values.numel() == 0:  # Пропускаем пустые батчи
                continue
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            # Очищаем память
            del input_values, attention_mask, labels, outputs, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Эпоха {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_values, attention_mask, labels in test_loader:
            if input_values.numel() == 0:  # Пропускаем пустые батчи
                continue
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_values, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Очищаем память
            del input_values, attention_mask, labels, outputs, preds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nAccuracy: {accuracy:.4f}")
    return model


def main():
    data, category_map = load_data('available_youtube.csv')
    print(f"Загружено {len(data)} записей. Категории: {category_map}")

    while True:
        try:
            train_size = int(input("Количество видео для обучения: "))
            test_size = int(input("Количество видео для тестирования: "))
            if train_size <= 0 or test_size <= 0:
                print("Число должно быть положительным")
            elif train_size + test_size > len(data):
                print(f"Превышено доступное количество записей, сумма данных для теста и тренировки не должна превосходить колличество ({len(data)})")
            else:
                break
        except ValueError:
            print("Введите целое число")

    # Выбираем случайные данные
    sampled_data = data.sample(n=train_size + test_size)
    train_data, test_data = train_test_split(sampled_data, test_size=test_size)
    train_links = train_data['link'].tolist()
    train_labels = train_data['label'].tolist()
    test_links = test_data['link'].tolist()
    test_labels = test_data['label'].tolist()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base-960h",
        num_labels=len(category_map)
    ).to(device)

    # Устанавливаем max_length (5 секунд при 16 кГц)
    max_length = 80000
    train_dataset = YouTubeAudioDataset(train_links, train_labels, processor, max_length=max_length)
    test_dataset = YouTubeAudioDataset(test_links, test_labels, processor, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Начинаем с 1 для отладки
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Начинаем с 1 для отладки
        collate_fn=collate_fn
    )

    trained_model = train_and_evaluate(train_loader, test_loader, model)

    del trained_model, train_dataset, test_dataset, train_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()