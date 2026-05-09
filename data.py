import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_NUM_WORKERS = min(4, os.cpu_count() or 1)
_PIN_MEMORY  = torch.cuda.is_available()


class RetinalDataset(Dataset):
    """
    Dataset per immagini retinali e risposte neurali.

    Supporta dataset expansion tramite aug_factor:
      aug_factor = 1 → solo campioni originali (nessuna copia aumentata)
      aug_factor = k → N originali + (k-1)*N copie augmentate = k*N campioni

    Le copie augmentate (idx >= N_orig) ricevono:
      - rumore gaussiano sull'immagine  (se img_noise_sigma > 0)
      - Poisson resample delle risposte (se poisson_resample = True)
    I campioni originali restano puliti.
    """

    def __init__(self, images, responses, img_noise_sigma=0.0, poisson_resample=False, aug_factor=1):
        """
        images           : numpy array (N, 108, 108)
        responses        : numpy array (N, 41)
        img_noise_sigma  : std del rumore gaussiano nelle copie augmentate (0 = disabilitato)
        poisson_resample : se True applica Poisson resample alle risposte nelle copie augmentate
        aug_factor       : fattore moltiplicativo del dataset (1 = nessuna espansione)
        """
        self.images = torch.tensor(
            images[:, np.newaxis, :, :],
            dtype=torch.float32
        )
        self.responses = torch.tensor(
            responses,
            dtype=torch.float32
        )
        self.img_noise_sigma  = img_noise_sigma
        self.poisson_resample = poisson_resample
        self.aug_factor       = aug_factor
        self._n_orig          = self.images.shape[0]

    def __len__(self):
        return self._n_orig * self.aug_factor

    def __getitem__(self, idx):
        orig_idx = idx % self._n_orig
        image    = self.images[orig_idx]
        response = self.responses[orig_idx]

        # augmentation applicata solo alle copie extra (non agli originali)
        if idx >= self._n_orig:
            if self.img_noise_sigma > 0.0:
                image = image + torch.randn_like(image) * self.img_noise_sigma
            if self.poisson_resample:
                response = torch.poisson(response)

        return {
            "image"   : image,    # [1, 108, 108]
            "response": response, # [41]
        }


def get_dataloaders(npz_path, batch_size=32, img_noise_sigma=0.0, poisson_resample=False, aug_factor=1):
    """
    Carica il file .npz e restituisce tre DataLoader:
    train, validation, test
    """
    data = np.load(npz_path)

    # estrai e pulisci le immagini
    images_train = data['images_train'].squeeze(-1)  # (2910, 108, 108)
    images_val   = data['images_val'].squeeze(-1)    # (250,  108, 108)
    images_test  = data['images_test'].squeeze(-1)   # (30,   108, 108)

    # estrai le risposte
    responses_train = data['responses_train']        # (2910, 41)
    responses_val   = data['responses_val']          # (250,  41)

    # per il test le risposte sono (30_ripetizioni, 30_immagini, 41_cellule)
    # facciamo la media sulle ripetizioni → (30, 41)
    responses_test = data['responses_test'].mean(axis=0)  # (30, 41)

    # crea i dataset
    train_dataset = RetinalDataset(images_train, responses_train,
                                   img_noise_sigma=img_noise_sigma,
                                   poisson_resample=poisson_resample,
                                   aug_factor=aug_factor)
    val_dataset   = RetinalDataset(images_val,   responses_val)
    test_dataset  = RetinalDataset(images_test,  responses_test)

    _kw = dict(num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY,
               persistent_workers=_NUM_WORKERS > 0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **_kw)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **_kw)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **_kw)

    return train_loader, val_loader, test_loader


# test rapido — esegui questo file direttamente per verificare
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        "PNAS_paper_sorted_data.npz",
        batch_size=32
    )

    # prendi il primo batch e stampa le dimensioni
    batch = next(iter(train_loader))
    print("image shape   :", batch["image"].shape)    # [32, 1, 108, 108]
    print("response shape:", batch["response"].shape) # [32, 41]
    print("Train batches :", len(train_loader))
    print("Val batches   :", len(val_loader))
    print("Test batches  :", len(test_loader))