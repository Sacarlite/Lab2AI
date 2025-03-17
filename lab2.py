import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import yt_dlp
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import gc
import shutil
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['link', 'category'])
    categories = df['category'].unique()
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    df['label'] = df['category'].map(category_to_idx)
    print("Распределение классов:", df['category'].value_counts().to_dict())
    return df, category_to_idx


def download_and_load_audio(url, output_dir="temp_audio", ffmpeg_path="ffmpeg/bin/ffmpeg.exe"):
    unique_id = url.split('=')[-1]  # Extract video ID from URL for uniqueness
    cache_path = os.path.join(output_dir, f"{unique_id}.pt")
    output_file = os.path.join(output_dir, f"audio_{unique_id}.wav")

    if os.path.exists(cache_path):
        waveform, sample_rate = torch.load(cache_path)
        return waveform, sample_rate

    try:
        Path(output_dir).mkdir(exist_ok=True)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_file.replace('.wav', ''),  # Unique filename without extension
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192'
            }],
            'ffmpeg_location': ffmpeg_path,
            'nopart': True,  # Avoid .part files by writing directly to final file
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Скачиваем аудио с {url}")
            ydl.download([url])

        waveform, sample_rate = torchaudio.load(output_file)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform / waveform.abs().max()

        if torch.any(torch.isnan(waveform)) or torch.any(torch.isinf(waveform)):
            print(f"Warning: Invalid values in waveform from {url}")
            if os.path.exists(output_file):
                os.remove(output_file)
            return None, None

        # Save to cache and clean up
        torch.save((waveform, sample_rate), cache_path)
        if os.path.exists(output_file):
            os.remove(output_file)
        return waveform, sample_rate

    except Exception as e:
        print(f"Ошибка при обработке {url}: {str(e)}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return None, None


def extract_embeddings(links, labels, processor, model, all_data, used_indices, max_chunk_length=160000, batch_size=4):
    model.eval()
    embeddings = []
    valid_labels = []
    remaining_data = all_data.drop(used_indices)

    def process_audio(link_label):
        link, label = link_label
        waveform, sample_rate = download_and_load_audio(link)
        if waveform is None and len(remaining_data) > 0:
            replacement = remaining_data.sample(n=1).iloc[0]
            link, label = replacement['link'], replacement['label']
            waveform, sample_rate = download_and_load_audio(link)
        return waveform, sample_rate, label, link

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_audio, zip(links, labels)))

    for idx, (waveform, sample_rate, label, link) in enumerate(results):
        if waveform is None:
            print(f"Не удалось обработать {link}")
            continue

        waveform_np = waveform.squeeze().numpy()
        audio_length = len(waveform_np)
        chunks = [waveform_np[i:i + max_chunk_length] for i in range(0, audio_length, max_chunk_length)]

        chunk_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            processed = processor(
                batch_chunks,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_chunk_length
            )
            input_values = processed.input_values.to(device)
            with torch.no_grad():
                outputs = model(input_values)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            chunk_embeddings.extend(embedding)
            del input_values, outputs

        final_embedding = np.mean(chunk_embeddings, axis=0)
        embeddings.append(final_embedding)
        valid_labels.append(label)
        print(f"Обработано видео {idx + 1}/{len(links)}, длина аудио: {audio_length} сэмплов")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    embeddings = np.vstack(embeddings)
    return embeddings, valid_labels


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_mlp_classifier(model, train_loader, test_loader, epochs=3, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    best_accuracy = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        accuracy = accuracy_score(y_true, y_pred)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy * 100:.2f}%")

        scheduler.step(total_loss)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    print(f"Лучшая точность классификатора: {best_accuracy * 100:.2f}%")
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

    sampled_data = data.sample(n=train_size + test_size, random_state=42)
    links = sampled_data['link'].tolist()
    labels = sampled_data['label'].tolist()
    used_indices = sampled_data.index

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

    embeddings, valid_labels = extract_embeddings(links, labels, processor, model, data, used_indices)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, valid_labels, train_size=train_size,
                                                        test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = EmbeddingsDataset(X_train, y_train)
    test_dataset = EmbeddingsDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    num_classes = len(category_map)
    mlp_model = MLPClassifier(input_dim=X_train.shape[1], hidden_dim=256, num_classes=num_classes).to(device)
    train_mlp_classifier(mlp_model, train_loader, test_loader)

    if os.path.exists("temp_audio"):
        shutil.rmtree("temp_audio")

    del model, processor, train_dataset, test_dataset, train_loader, test_loader, mlp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()