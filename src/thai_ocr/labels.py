"""The 44 Thai consonants, in the order used by the dataset file names (01_ko_kai.png ... 44_ho_nokhuk.png)."""

CONSONANTS = [
    ("ก", "ko_kai"), ("ข", "kho_khai"), ("ฃ", "kho_khuat"), ("ค", "kho_khwai"),
    ("ฅ", "kho_khon"), ("ฆ", "kho_rakhang"), ("ง", "ngo_ngu"), ("จ", "cho_chan"),
    ("ฉ", "cho_ching"), ("ช", "cho_chang"), ("ซ", "so_so"), ("ฌ", "cho_choe"),
    ("ญ", "yo_ying"), ("ฎ", "do_chada"), ("ฏ", "to_patak"), ("ฐ", "tho_than"),
    ("ฑ", "tho_montho"), ("ฒ", "tho_phuthao"), ("ณ", "no_nen"), ("ด", "do_dek"),
    ("ต", "to_tao"), ("ถ", "tho_thung"), ("ท", "tho_thahan"), ("ธ", "tho_thong"),
    ("น", "no_nu"), ("บ", "bo_baimai"), ("ป", "po_pla"), ("ผ", "pho_phung"),
    ("ฝ", "fo_fa"), ("พ", "pho_phan"), ("ฟ", "fo_fan"), ("ภ", "pho_samphao"),
    ("ม", "mo_ma"), ("ย", "yo_yak"), ("ร", "ro_rua"), ("ล", "lo_ling"),
    ("ว", "wo_waen"), ("ศ", "so_sala"), ("ษ", "so_rusi"), ("ส", "so_suea"),
    ("ห", "ho_hip"), ("ฬ", "lo_chula"), ("อ", "o_ang"), ("ฮ", "ho_nokhuk"),
]

NUM_CLASSES = len(CONSONANTS)
THAI = [c for c, _ in CONSONANTS]
NAMES = [n for _, n in CONSONANTS]


def label_from_filename(filename: str) -> int:
    """'07_ngo_ngu.png' -> 6. Uses the number prefix; raises ValueError if it is missing or out of range."""
    stem = filename.rsplit("/", 1)[-1].split(".")[0]
    prefix = stem.split("_", 1)[0]
    idx = int(prefix) - 1
    if not 0 <= idx < NUM_CLASSES:
        raise ValueError(f"{filename}: letter number {prefix} is not 01-44")
    return idx
