"""DiaMED dataset adapter for ICD-10 chapter classification."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.classification import ClassificationTask

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
        """Load train/val/test splits from HuggingFace."""
        hf_ds = load_dataset("DrBenchmark/DiaMED", trust_remote_code=True)

        splits: dict[str, Dataset] = {}
        split_map = {"train": "train", "validation": "val", "test": "test"}

        for hf_split, local_split in split_map.items():
            if hf_split not in hf_ds:
                continue

            records = []
            for item in hf_ds[hf_split]:
                text = item.get("clinical_case", "")
                label_idx = item.get("icd-10")

                # Handle None text
                if text is None:
                    text = ""
                text = text.strip()
                if not text:
                    continue

                # Convert integer label index to chapter name
                if label_idx is None:
                    continue
                if isinstance(label_idx, int):
                    if 0 <= label_idx < len(ICD10_CHAPTERS):
                        label = ICD10_CHAPTERS[label_idx]
                    else:
                        continue
                elif isinstance(label_idx, str):
                    label = label_idx.strip()
                    if not label:
                        continue
                else:
                    continue

                records.append({
                    "text": text,
                    "label": label,
                })

            if records:
                splits[local_split] = Dataset.from_list(records)

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
