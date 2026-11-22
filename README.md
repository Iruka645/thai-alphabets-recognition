# Thai handwritten consonant recognition (ก–ฮ)

Give it a picture of one handwritten Thai consonant and it tells you which of the 44 letters it is.
It is built to work on **handwriting from people who are not in the dataset**.

## The idea in plain words

- **Each person writes each letter once.** About 20 people means only about 20 examples per letter.
  That is very little, so the setup is built around that.
- **We test on people, never on pictures.** If one person's letters end up in both training and
  testing, the score looks great but is fake. So we always hold out whole people.
- **Every person gets a turn as "the stranger."** Writers are split into 5 groups. We train 5 times,
  each time leaving one group out and testing on it. Together, the 5 rounds score everyone with a
  model that never saw their handwriting. With so few people, this is much more trustworthy than
  one fixed test group of 3.
- **We fake extra handwriting styles.** During training, each letter is randomly tilted, stretched,
  wobbled, and drawn with a thicker or thinner pen (see `python -m thai_ocr.preview`). It is never
  mirrored, because a flipped letter can look like a different letter.
- **The model is small on purpose.** It is a small CNN with about 0.8M parameters, sized for
  32×32 images and about 900 examples. A bigger model would mostly memorise.
- **More people beats a fancier model.** If the score is too low, handwriting from 10 more people
  will probably help more than any change to the code.

## Setup (Windows desktop with RTX 3060 Ti)

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate

# 1. PyTorch with GPU support: copy the exact command for your system from https://pytorch.org/get-started/locally/
#    It looks like:  pip install torch --index-url https://download.pytorch.org/whl/cu126
# 2. This project
pip install -e ".[dev]"

python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

## Data

Put **one zip or one folder per person** into `data/raw/`. Name the files `01_ko_kai.png` …
`44_ho_nokhuk.png`, 32×32 grayscale, dark ink on white, the same as the cleaned sample. Sub-folders
inside a zip are fine.

```
data/raw/
  person_a.zip            ← zip name = writer ID
  person_b.zip
  person_c/               ← a plain folder also works
    01_ko_kai.png ... 44_ho_nokhuk.png
```

`data/` is git-ignored, so people's handwriting never gets pushed to GitHub.

## Workflow

```bash
# 1. Collect everything into one file and check that no one is missing letters
python -m thai_ocr.prepare

# 2. (optional) Check that the fake handwriting variations still look realistic
python -m thai_ocr.preview                    # -> runs/augment_preview.png

# 3. Get the honest score: how well does it read a stranger's handwriting?
python -m thai_ocr.train cv                   # -> runs/cv-<time>/report.txt, confusion.png

# 4. Happy with the score? Train on everyone and save the model you will use
python -m thai_ocr.train final                # -> runs/final-<time>/model.pt

# 5. Use it on any image (big photos are cleaned and cropped automatically)
python -m thai_ocr.predict runs/final-<time>/model.pt my_letter.jpg
```

On a 3060 Ti, a full `cv` run (5 × 150 epochs) should take a few minutes.

## Reading the results

`report.txt` shows:
- **Overall accuracy on unseen writers.** This is the number that matters.
- **Per writer.** If one person scores much lower, their style is unusual, or their data has a
  problem, such as a letter saved under the wrong number.
- **Hardest letters and most common mix-ups.** Expect look-alike pairs such as ด/ต, บ/ป, ผ/ฝ, พ/ฟ,
  ช/ซ, ฎ/ฏ, ข/ฃ and ค/ฅ. ฃ and ฅ are rarely written today, so they are often the weakest.

## Things to try (always compare the `cv` score before and after)

| Change | Flag |
|---|---|
| Stronger or weaker fake variation | `--aug 1.3` / `--aug 0.7` |
| Train longer or shorter | `--epochs 300` |
| Bigger or smaller model | `--width 48` / `--width 24` |
| More or less dropout | `--dropout 0.5` |
| Check the score isn't luck | run again with `--seed 1`, `--seed 2` |

Because results with 20 people move around by a few percent from run to run, only trust an
improvement that holds up across a couple of seeds.

## Project layout

```
src/thai_ocr/
  labels.py      44 letters and their file-name numbers
  preprocess.py  cleaning (for new images) + normalisation
  prepare.py     data/raw  ->  data/processed/dataset.npz
  model.py       the CNN + augmentation
  train.py       `cv` (honest score) and `final` (model to use)
  report.py      accuracy tables and confusion chart
  predict.py     classify new images
  preview.py     picture of what augmentation does
tests/           quick checks: python -m pytest
```
