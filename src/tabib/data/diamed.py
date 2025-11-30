"""DiaMED dataset adapter for ICD-10 chapter classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.classification import ClassificationTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "drbenchmark" / "data" / "drbenchmark" / "diamed" / "splits"

# ICD-10 chapters (22 classes)
ICD10_CHAPTERS = [
    'A00-B99  Certain infectious and parasitic diseases',
    'C00-D49  Neoplasms',
    'D50-D89  Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
    'E00-E89  Endocrine, nutritional and metabolic diseases',
    'F01-F99  Mental, Behavioral and Neurodevelopmental disorders',
    'G00-G99  Diseases of the nervous system',
    'H00-H59  Diseases of the eye and adnexa',
    'H60-H95  Diseases of the ear and mastoid process',
    'I00-I99  Diseases of the circulatory system',
    'J00-J99  Diseases of the respiratory system',
    'K00-K95  Diseases of the digestive system',
    'L00-L99  Diseases of the skin and subcutaneous tissue',
    'M00-M99  Diseases of the musculoskeletal system and connective tissue',
    'N00-N99  Diseases of the genitourinary system',
    'O00-O9A  Pregnancy, childbirth and the puerperium',
    'P00-P96  Certain conditions originating in the perinatal period',
    'Q00-Q99  Congenital malformations, deformations and chromosomal abnormalities',
    'R00-R99  Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
    'S00-T88  Injury, poisoning and certain other consequences of external causes',
    'U00-U85  Codes for special purposes',
    'V00-Y99  External causes of morbidity',
    'Z00-Z99  Factors influencing health status and contact with health services',
]


class DiaMEDAdapter(DatasetAdapter):
    """Adapter for DiaMED ICD-10 chapter classification.

    Clinical cases from French medical literature classified by ICD-10 chapter.
    """

    def __init__(self) -> None:
        self._label_vocab: list[str] = ICD10_CHAPTERS

    @property
    def name(self) -> str:
        return "diamed"

    def load_splits(self) -> dict[str, Dataset]:
        if not DATA_DIR.exists():
            raise FileNotFoundError(
                f"DiaMED data not found. Expected at {DATA_DIR}. "
                "Download from DrBenchmark/DiaMED on HuggingFace."
            )

        splits: dict[str, Dataset] = {}

        for split_name, filename in [("train", "train.json"), ("val", "validation.json"), ("test", "test.json")]:
            filepath = DATA_DIR / filename
            if not filepath.exists():
                continue

            with filepath.open(encoding="utf-8") as f:
                data = json.load(f)

            records = []
            for item in data:
                text = item.get("clinical_case", "")
                label = item.get("icd-10", "")

                # Handle None or non-string values
                if text is None:
                    text = ""
                if label is None or not isinstance(label, str):
                    continue

                text = text.strip()
                label = label.strip()

                if not text or not label:
                    continue

                records.append({
                    "text": text,
                    "label": label,
                })

            if records:
                splits[split_name] = Dataset.from_list(records)

        return splits

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, ClassificationTask):
            raise ValueError(
                f"DiaMED expects ClassificationTask, got {type(task)}"
            )

        task.ensure_labels(self._label_vocab)
        label_map = task.label_space

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            label_id = label_map.get(example["label"], 0)
            return {
                "text": example["text"],
                "labels": label_id,
            }

        processed = dataset.map(map_example)
        processed.set_format(type="python")
        return processed
