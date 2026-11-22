import zipfile

import numpy as np
import torch
from PIL import Image, ImageDraw

from thai_ocr.labels import NAMES, NUM_CLASSES, label_from_filename
from thai_ocr.model import LetterCNN, augment
from thai_ocr.preprocess import clean_image, to_array


def _letter_png(size=32, offset=0):
    img = Image.new("L", (size, size), 255)
    ImageDraw.Draw(img).ellipse([6 + offset, 6, size - 6, size - 6], outline=0, width=3)
    return img


def test_labels():
    assert NUM_CLASSES == 44
    assert label_from_filename("01_ko_kai.png") == 0
    assert label_from_filename("some/dir/44_ho_nokhuk.png") == 43
    for i, n in enumerate(NAMES):
        assert label_from_filename(f"{i + 1:02d}_{n}.png") == i


def test_clean_image_handles_big_inverted_photo():
    big = Image.new("L", (300, 200), 20)  # dark background, light ink
    ImageDraw.Draw(big).line([(100, 50), (180, 150)], fill=240, width=12)
    out = to_array(clean_image(big))
    assert out.shape == (32, 32)
    assert out[0, 0] == 255 and out.min() < 100  # white background, dark ink


def test_augment_keeps_shape_and_range():
    x = torch.rand(8, 1, 32, 32)
    y = augment(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    assert torch.equal(augment(x, 0), x)


def test_model_output_shape():
    assert LetterCNN()(torch.zeros(2, 1, 32, 32)).shape == (2, 44)


def test_prepare_reads_zips_and_folders(tmp_path, monkeypatch):
    from thai_ocr import prepare

    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "alice.zip", "w") as zf:
        for i, n in enumerate(NAMES):
            p = tmp_path / "tmp.png"
            _letter_png(offset=i % 5).save(p)
            zf.write(p, f"thai_consonants_32/{i + 1:02d}_{n}.png")
    (raw / "bob").mkdir()
    _letter_png().save(raw / "bob" / "01_ko_kai.png")

    out = tmp_path / "ds.npz"
    monkeypatch.setattr("sys.argv", ["prepare", "--raw", str(raw), "--out", str(out)])
    prepare.main()
    d = np.load(out)
    assert d["X"].shape == (45, 32, 32)
    assert list(d["writers"]) == ["alice", "bob"]
    assert (d["writer"] == 1).sum() == 1
