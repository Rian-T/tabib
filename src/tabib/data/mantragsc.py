"""MANTRAGSC dataset adapter for French biomedical NER."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from datasets import Dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.ner_span import NERSpanTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "drbenchmark" / "data" / "drbenchmark" / "mantragsc" / "GSC-v1.1"

# Entity group mapping (from UMLS semantic types)
ENTITY_GROUPS = {
    "ANAT": "anatomy",
    "CHEM": "chemical",
    "DEVI": "device",
    "DISO": "disorder",
    "GEOG": "geography",
    "LIVB": "living_being",
    "OBJC": "object",
    "PHEN": "phenomenon",
    "PHYS": "physiology",
    "PROC": "procedure",
}


class MANTRAGSCAdapter(DatasetAdapter):
    """Adapter for MANTRAGSC French biomedical NER dataset.

    Uses the Medline French manually annotated gold standard corpus.
    Entity types: anatomy, chemical, device, disorder, geography,
    living_being, object, phenomenon, physiology, procedure.
    """

    def __init__(self, source: str = "medline") -> None:
        """Initialize adapter.

        Args:
            source: Data source - 'medline', 'emea', or 'patent'
        """
        self._source = source.lower()

    @property
    def name(self) -> str:
        return f"mantragsc_{self._source}"

    def load_splits(self) -> dict[str, Dataset]:
        if not DATA_DIR.exists():
            raise FileNotFoundError(
                f"MANTRAGSC data not found. Expected at {DATA_DIR}. "
                "Download and extract GSC-v1.1.zip from DrBenchmark/MANTRAGSC."
            )

        # Map source to filename
        source_files = {
            "medline": "Medline_GSC_fr_man.xml",
            "emea": "EMEA_GSC_fr_man.xml",
            "patent": "Patent_GSC_fr_man.xml",
        }

        filename = source_files.get(self._source)
        if not filename:
            raise ValueError(f"Unknown source: {self._source}. Use 'medline', 'emea', or 'patent'")

        xml_path = DATA_DIR / filename
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        # Parse XML and extract documents
        records = self._parse_xml(xml_path)

        # No predefined train/val/test split in MANTRAGSC
        # Use 80/10/10 split
        n = len(records)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        return {
            "train": Dataset.from_list(records[:train_end]),
            "val": Dataset.from_list(records[train_end:val_end]),
            "test": Dataset.from_list(records[val_end:]),
        }

    def _parse_xml(self, xml_path: Path) -> list[dict[str, Any]]:
        """Parse MANTRAGSC XML format."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        records = []
        for doc in root.findall("document"):
            doc_id = doc.get("id", "")

            for unit in doc.findall("unit"):
                text_elem = unit.find("text")
                if text_elem is None or text_elem.text is None:
                    continue

                text = text_elem.text
                entities = []

                for ent in unit.findall("e"):
                    offset = int(ent.get("offset", 0))
                    length = int(ent.get("len", 0))
                    grp = ent.get("grp", "")
                    ent_text = ent.text or ""

                    # Map group to entity type
                    entity_type = ENTITY_GROUPS.get(grp, grp.lower())

                    entities.append({
                        "start": offset,
                        "end": offset + length,
                        "label": entity_type,
                        "text": ent_text,
                    })

                records.append({
                    "doc_id": doc_id,
                    "text": text,
                    "entities": entities,
                })

        return records

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, NERSpanTask):
            raise ValueError(
                f"MANTRAGSC expects NERSpanTask, got {type(task)}"
            )

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "text": example["text"],
                "spans": example["entities"],
            }

        processed = dataset.map(map_example)
        processed.set_format(type="python")
        return processed
