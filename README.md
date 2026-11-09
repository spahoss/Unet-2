# U-Net για κατάτμηση της zona σε εικόνες εμβρύων

Ερευνητικό project δυαδικής κατάτμησης εικόνων εμβρύων με U-Net. Υποστηρίζει εκπαίδευση σε όλες τις ηλικίες ή χωριστά στις **25h**, **69h** και **117h**, εκτέλεση σε CPU/CUDA, grouped train/validation/test splits, live διαγράμματα, model summary, πρόβλεψη από terminal και εφαρμογή Streamlit.

> Το project προορίζεται για πειραματική και ερευνητική χρήση. Δεν αποτελεί κλινικό εργαλείο.

## Μοντέλο

Χρησιμοποιείται `smp.Unet` από το `segmentation-models-pytorch` με ResNet-18 encoder, RGB είσοδο, μία κλάση εξόδου, ImageNet normalization, Adam optimizer και BCE + Dice loss. Η προεπιλεγμένη είσοδος είναι 512×512 με letterbox padding. Το padding εξαιρείται από loss και μετρικές.

## Κύρια αρχεία

| Αρχείο | Ρόλος |
|---|---|
| `train.py` | Έλεγχος δεδομένων, training, validation, test και αποθήκευση αποτελεσμάτων |
| `predict.py` | Prediction από αποθηκευμένο `best_model.pt` |
| `data_utils.py` | Έλεγχος ζευγών, grouped splits και preprocessing |
| `emvryos_data.py` | Ανάγνωση MATLAB v5 `gTruth.mat` |
| `age_utils.py` | Επιλογή ηλικίας και έλεγχος checkpoint |
| `prepare_data.py` | Δημιουργία dataset με σταθερά splits |
| `training_progress.py` | Live και αποθηκευμένα training charts |
| `model_summary.py` | Πίνακας layers, activations και parameters |
| `streamlit_app.py` | Γραφική εφαρμογή prediction |
| `web_predict.py` | Validation uploads και prediction backend |

## Ελεγμένη διαμόρφωση

- Ubuntu ή Ubuntu μέσω WSL2
- Python 3.13
- PyTorch 2.7.1
- CUDA 12.6
- NVIDIA GeForce GTX 1080

Το project λειτουργεί και σε CPU, αλλά το training είναι σημαντικά πιο αργό.

## 1. Clone και εγκατάσταση

```bash
git clone https://github.com/spahoss/Unet-2.git
cd Unet-2

sudo apt update
sudo apt install pipx
pipx install uv
pipx ensurepath
source ~/.bashrc

uv python install 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate

UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 uv pip install \
  -r requirements.txt \
  --index-strategy unsafe-best-match
```

Η πρώτη εγκατάσταση είναι μεγάλη επειδή περιλαμβάνει τα CUDA wheels του PyTorch.

### Έλεγχος GPU

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Αναμενόμενο αποτέλεσμα:

```text
PyTorch: 2.7.1+cu126
CUDA: 12.6
GPU available: True
GPU: NVIDIA GeForce GTX 1080
```

Στο WSL2 ο NVIDIA driver εγκαθίσταται στα Windows. Μην εγκαταστήσεις Linux display driver μέσα στο WSL.

## 2. Δεδομένα

Η προεπιλεγμένη διαδρομή είναι:

```text
/mnt/c/Users/spaho/Desktop/Emvryos
```

Σε άλλη τοποθεσία χρησιμοποίησε `--data-root`. Ο αναγνώστης αναζητά αναδρομικά τα `gTruth.mat` κάτω από `embrios1` έως `embrios4` και χρησιμοποιεί τις αποθηκευμένες MATLAB λίστες για την αντιστοίχιση εικόνων/μασκών.

Το πλήρες dataset περιλαμβάνει 114 εικόνες από 38 έμβρυα:

| Πηγή | Εικόνες | Έμβρυα |
|---|---:|---:|
| embryos1 | 39 | 13 |
| embryos2 | 24 | 8 |
| embryos3 | 24 | 8 |
| embryos4 | 27 | 9 |
| **Σύνολο** | **114** | **38** |

Οι λήψεις 25h, 69h και 117h του ίδιου εμβρύου παραμένουν στο ίδιο split. Το `embrios3` περιέχει contour annotations, ενώ οι άλλες πηγές κυρίως area annotations. Η προεπιλογή τα περιλαμβάνει όλα. Για εξαίρεση χρησιμοποίησε `--exclude-contour-source`.

## 3. Έλεγχος dataset

```bash
python train.py \
  --data-root "/mnt/c/Users/spaho/Desktop/Emvryos" \
  --check
```

Ελέγχονται paths, IDs, hashes, διαστάσεις, τιμές labels, μη κενές μάσκες, grouped splits και resized masks. Δεν εκτελείται training.

## 4. Training

### Όλες οι ηλικίες

```bash
python train.py \
  --data-root "/mnt/c/Users/spaho/Desktop/Emvryos" \
  --epochs 30 \
  --batch-size 2 \
  --device cuda \
  --seed 42 \
  --run-name zona_all_01
```

### Μία ηλικία

```bash
python train.py \
  --data-root "/mnt/c/Users/spaho/Desktop/Emvryos" \
  --age 25 \
  --epochs 30 \
  --batch-size 2 \
  --device cuda \
  --seed 42 \
  --run-name zona_25h_01
```

Έγκυρες ηλικίες: `25`, `69`, `117`. Χωρίς `--age` χρησιμοποιούνται όλες. Για age-specific training το split seed είναι `seed + age`. Το αναμενόμενο split κάθε ηλικίας είναι 24 train / 9 validation / 5 test εικόνες. Για όλες τις ηλικίες είναι 72/27/15.

### Live charts και model summary

```bash
python train.py \
  --data-root "/mnt/c/Users/spaho/Desktop/Emvryos" \
  --age 25 \
  --epochs 30 \
  --device cuda \
  --run-name zona_25h_monitor_01 \
  --plot \
  --summary
```

Με `--plot` εμφανίζεται σύνδεσμος `Live training charts:`. Άνοιξέ τον όσο εκτελείται το training. Το `training_progress.html` αποθηκεύεται πάντα. Με `--summary` αποθηκεύονται και `model_summary.html/csv/json`.

Μόνο model summary:

```bash
python train.py \
  --data-root "/mnt/c/Users/spaho/Desktop/Emvryos" \
  --age 25 \
  --device cpu \
  --weights none \
  --summary-only \
  --run-name layer_analysis_25h
```

### Βασικές επιλογές

| Επιλογή | Default | Περιγραφή |
|---|---:|---|
| `--epochs` | 30 | Μέγιστος αριθμός epochs |
| `--batch-size` | 2 | Batch size |
| `--size` | 512 | Τουλάχιστον 64 και πολλαπλάσιο του 32 |
| `--lr` | 0.0001 | Learning rate |
| `--weights` | `imagenet` | `imagenet` ή `none` |
| `--device` | `cpu` | `cpu`, `cuda` ή `auto` |
| `--patience` | 10 | Early-stopping patience |
| `--seed` | 42 | Training/split base seed |

Το `--weights imagenet` μπορεί να κατεβάσει pretrained weights την πρώτη φορά. Για offline χρήση βάλε `--weights none`. Αν εξαντλείται η GPU μνήμη, μείωσε πρώτα το batch size και μετά το input size.

## 5. Αποτελέσματα training

Κάθε run γράφεται στο `runs/<run-name>/`:

```text
runs/<run-name>/
├── best_model.pt
├── config.json
├── history.csv
├── split.json
├── test_metrics.json
├── training_progress.html
├── training_progress.json
├── model_summary.html       # όταν ζητηθεί
├── model_summary.csv        # όταν ζητηθεί
├── model_summary.json       # όταν ζητηθεί
└── test_predictions/
```

Το καλύτερο checkpoint επιλέγεται με validation Dice. Το held-out test αξιολογείται μία φορά μετά την επιλογή checkpoint. Το pixel accuracy περιλαμβάνει το φόντο και δεν πρέπει να ερμηνεύεται μόνο του.

## 6. Prediction από terminal

### Βασική χρήση

```bash
python predict.py \
  --checkpoint runs/zona_25h_01/best_model.pt \
  --image "/διαδρομή/προς/εικόνα.png" \
  --age 25 \
  --device cuda
```

Το `--age` ελέγχει τη συμβατότητα του checkpoint. Αν το filename περιέχει ηλικία όπως `25h`, ελέγχεται επίσης.

### Threshold, border removal και output

```bash
python predict.py \
  --checkpoint runs/zona_25h_01/best_model.pt \
  --image "/διαδρομή/προς/εικόνα.png" \
  --threshold 0.60 \
  --border-ratio 0.10 \
  --output predictions/example_01 \
  --device cuda
```

Το output directory δεν πρέπει να υπάρχει ήδη. Χωρίς `--output`, δημιουργείται timestamped φάκελος στο `predictions/`.

### Με ground-truth mask

```bash
python predict.py \
  --checkpoint runs/zona_25h_01/best_model.pt \
  --image "/διαδρομή/προς/εικόνα.png" \
  --mask "/διαδρομή/προς/mask.png" \
  --age 25 \
  --device cuda
```

Στο CLI η reference mask πρέπει να είναι ενός καναλιού, με ίδιες διαστάσεις και τιμές ακριβώς `0/1`.

Κάθε prediction περιέχει `prediction_raw.png`, `prediction.png`, `probability.png`, `overlay.png`, `comparison.png` και, όταν υπάρχει reference mask, `metrics.json`.

## 7. Streamlit εφαρμογή

Χρειάζεται τουλάχιστον ένα `best_model.pt` μέσα σε `runs/<run-name>/` ή χειροκίνητη διαδρομή checkpoint.

```bash
python -m streamlit run streamlit_app.py
```

Η εφαρμογή επιτρέπει επιλογή ηλικίας, checkpoint, CPU/CUDA και threshold. Δέχεται εικόνες PNG/JPG/TIFF έως 20 MB και 12 εκατομμύρια pixels. Η προαιρετική reference mask μπορεί να έχει τιμές `0/1` ή `0/255`. Τα αποτελέσματα προβάλλονται ως mask, overlay και comparison και μπορούν να ληφθούν ως PNG ή ZIP.

Το checkpoint πρέπει να έχει εκπαιδευτεί για την επιλεγμένη ηλικία.

## 8. Προετοιμασμένο dataset

```bash
python prepare_data.py \
  --source "/mnt/c/Users/spaho/Desktop/Emvryos" \
  --destination data_emvryos \
  --seed 42
```

Δημιουργούνται `train/images`, `train/masks`, `val/images`, `val/masks`, `test/images`, `test/masks` και `dataset.json`. Ο προορισμός δεν πρέπει να υπάρχει ήδη.

```bash
python train.py --data-root data_emvryos --check
python train.py --data-root data_emvryos --epochs 30 --device cuda --run-name prepared_01
```

Τα αποθηκευμένα splits του manifest διατηρούνται.

## 9. GPU και περιβάλλον

Παρακολούθηση GPU:

```bash
watch -n 1 nvidia-smi
```

Ενεργοποίηση περιβάλλοντος σε νέο terminal:

```bash
cd ~/Unet-2
source .venv/bin/activate
```

Έξοδος:

```bash
deactivate
```

## Περιορισμοί

- Το test set είναι μικρό και οι μετρικές αποτελούν αρχική εκτίμηση.
- Τα contour annotations του `embrios3` δεν έχουν τον ίδιο ορισμό στόχου με τα area annotations.
- Το threshold πρέπει να επιλέγεται με validation δεδομένα, όχι με το held-out test.
- Το μοντέλο κάνει segmentation της επισημασμένης περιοχής· δεν ταξινομεί ποιότητα ή βιωσιμότητα εμβρύου.
- Δεν υπάρχει resume training από checkpoint.

## Git

Τα `.venv/`, `runs/`, `predictions/`, `.env`, `__pycache__/` και `*.pyc` αγνοούνται από το Git. Τα checkpoints και τα νέα generated predictions δεν ανεβαίνουν αυτόματα.
