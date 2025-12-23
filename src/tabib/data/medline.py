"""MEDLINE dataset adapter."""

from pathlib import Path

from tabib.data.brat import BRATDatasetAdapter


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "MEDLINE"


class MEDLINEAdapter(BRATDatasetAdapter):
    """Adapter for the MEDLINE dataset in BRAT format."""

    def __init__(self, data_dir: str | Path | None = None):
        super().__init__(
            data_dir=data_dir if data_dir else DEFAULT_DATA_DIR,
            name="medline",
            filter_nested=True,  # Keep coarsest granularity (CamemBERT-bio methodology)
        )

