"""EMEA dataset adapter."""

from pathlib import Path

from tabib.data.brat import BRATDatasetAdapter


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "EMEA"


class EMEAAdapter(BRATDatasetAdapter):
    """Adapter for the EMEA dataset in BRAT format.

    Uses line-by-line splitting by default since EMEA documents are very long
    (25K+ chars) but there are only 11 train / 15 test documents.
    Each line becomes one sample for better NER performance.
    """

    def __init__(self, data_dir: str | Path | None = None, chunk_size: int = -1):
        """Initialize EMEA adapter with line-by-line splitting.

        Args:
            data_dir: Path to EMEA data
            chunk_size: -1 for line-by-line (default), 0 for no chunking,
                        >0 for character-based chunks
        """
        super().__init__(
            data_dir=data_dir if data_dir else DEFAULT_DATA_DIR,
            name="emea",
            chunk_size=chunk_size
        )

