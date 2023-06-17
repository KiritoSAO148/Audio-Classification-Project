import torch
import torchaudio
import pandas as pd
from torch.utils.data import random_split
import itertools
from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset

label_mapping = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}

def predict(model, test_dl, device, label_mapping):
    num_items_test = len(test_dl.dataset)
    print("Độ dài của test_dl:", num_items_test)
    correct_prediction = 0
    total_prediction = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dl):
            inputs, labels = inputs.to(device), labels.to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            outputs = model(inputs)

            _, prediction = torch.max(outputs, 1)

            predicted_labels = [label_mapping[p.item()] for p in prediction]
            actual_labels = [label_mapping[l.item()] for l in labels]

            audio_paths = [df.loc[index, 'relative_path'] for index in
                           test_dl.dataset.indices[i * test_dl.batch_size: (i + 1) * test_dl.batch_size]]

            audio_info = zip(audio_paths, predicted_labels, actual_labels)
            audio_info = itertools.islice(audio_info, len(inputs))  # Chỉ lấy số lượng file tương ứng với batch hiện tại

            for audio_path, predicted_label, actual_label in audio_info:
                print(f"i: {i + 1}, Audio Path: {audio_path}, Predicted: {predicted_label}, Actual: {actual_label}")

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

if __name__ == "__main__":
    data_path = 'UrbanSound8K'
    metadata_file = data_path + '/metadata/UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df.head()

    myds = UrbanSoundDataset(df, data_path)

    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    cnn = CNNNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(device)
    cnn.load_state_dict(torch.load('cnnnet.pt'))
    cnn.eval()

    predict(cnn, val_dl, device, label_mapping)