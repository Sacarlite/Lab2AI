import pandas as pd
import torch
import torchaudio
import yt_dlp
import os
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
import gc
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['link', 'category'])
    categories = df['category'].unique()
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    df['label'] = df['category'].map(category_to_idx)
    return df, category_to_idx


def check_video_availability(url):
    """Проверка доступности видео перед загрузкой"""
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            ydl.extract_info(url, download=False)
        return True
    except Exception:
        print(f"Видео по ссылке {url} недоступно для скачивания")
        return False


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
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        elif waveform.shape[1] < max_length:
            padding = torch.zeros(waveform.shape[0], max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)

        os.remove(audio_file)
        return waveform, sample_rate
    except Exception as e:
        print(f"Ошибка при обработке {url}: {str(e)}")
        return None, None


class YouTubeAudioDataset(Dataset):
    def __init__(self, links, labels, processor, max_length=80000):
        self.links = [link for link in links if check_video_availability(link)]
        self.labels = [labels[i] for i, link in enumerate(links) if check_video_availability(link)]
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        link = self.links[idx]
        label = self.labels[idx]
        waveform, sample_rate = download_and_load_audio(link, max_length=self.max_length)
        if waveform is None:
            return (
                torch.zeros(1, self.max_length),
                torch.zeros(1, self.max_length),
                torch.tensor(label, dtype=torch.long)
            )

        waveform_np = waveform.squeeze().numpy()
        processed = self.processor(
            waveform_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        attention_mask = processed.get('attention_mask', torch.ones_like(processed.input_values))
        return (
            processed.input_values.squeeze(0),
            attention_mask.squeeze(0),
            torch.tensor(label, dtype=torch.long)
        )


def collate_fn(batch):
    filtered_batch = [item for item in batch if not torch.all(item[0] == 0)]
    if not filtered_batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    input_values = torch.stack([item[0] for item in filtered_batch])
    attention_masks = torch.stack([item[1] for item in filtered_batch])
    labels = torch.stack([item[2] for item in filtered_batch])

    return input_values, attention_masks, labels


def train_and_evaluate(train_loader, test_loader, model, num_epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_time = time.time()
    total_train_loss = 0
    total_test_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()

        for batch_idx, (input_values, attention_mask, labels) in enumerate(train_loader):
            if input_values.numel() == 0:
                continue
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

            del input_values, attention_mask, labels, outputs, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(train_loader)
        total_train_loss += avg_epoch_loss

        model.eval()
        epoch_preds = []
        epoch_labels = []
        with torch.no_grad():
            for input_values, attention_mask, labels in test_loader:
                if input_values.numel() == 0:
                    continue
                input_values = input_values.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(input_values, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())

                del input_values, attention_mask, labels, outputs, preds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        epoch_accuracy = np.mean(np.array(epoch_preds) == np.array(epoch_labels)) * 100
        total_test_accuracy += epoch_accuracy

        print(
            f"Эпоха {epoch + 1}, Время обучения: {round(time.time() - epoch_start_time, 2)}с., "
            f"Потери: {avg_epoch_loss:.4f}, Точность: {epoch_accuracy:.2f}%"
        )

    avg_train_loss = total_train_loss / num_epochs
    avg_test_accuracy = total_test_accuracy / num_epochs
    total_time = round(time.time() - start_time, 2)

    print("\nОбщая статистика:")
    print(
        f"Всего эпох: {num_epochs}, Общее время обучения: {total_time}с., "
        f"Средние потери: {avg_train_loss:.4f}, Средняя точность: {avg_test_accuracy:.2f}%"
    )

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
                print(f"Превышено доступное количество записей ({len(data)})")
            else:
                break
        except ValueError:
            print("Введите целое число")

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

    max_length = 80000
    train_dataset = YouTubeAudioDataset(train_links, train_labels, processor, max_length=max_length)
    test_dataset = YouTubeAudioDataset(test_links, test_labels, processor, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    trained_model = train_and_evaluate(train_loader, test_loader, model)

    # Очистка временной папки
    if os.path.exists("temp_audio"):
        shutil.rmtree("temp_audio")

    del trained_model, train_dataset, test_dataset, train_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()