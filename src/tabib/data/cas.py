"""CAS dataset adapters."""

from pathlib import Path

from tabib.data.brat import BRATDatasetAdapter


DEFAULT_CAS1_DIR = Path(__file__).resolve().parents[3] / "data" / "CAS1"
DEFAULT_CAS2_DIR = Path(__file__).resolve().parents[3] / "data" / "CAS2"


class CAS1Adapter(BRATDatasetAdapter):
    """Adapter for the CAS1 dataset stored in BRAT format.

    Uses line-by-line splitting by default for better NER performance.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        chunk_size: int = -1,
        filter_nested: bool = True,
    ):
        """Initialize CAS1 adapter with line-by-line splitting.

        Args:
            data_dir: Path to CAS1 data
            chunk_size: -1 for line-by-line (default), 0 for no chunking,
                        >0 for character-based chunks
            filter_nested: If True (default), keep only coarsest granularity
                          entities (matches CamemBERT-bio paper methodology)
        """
        super().__init__(
            data_dir=data_dir if data_dir else DEFAULT_CAS1_DIR,
            name="cas1",
            chunk_size=chunk_size,
            filter_nested=filter_nested,
        )


class CAS2Adapter(BRATDatasetAdapter):
    """Adapter for the CAS2 dataset stored in BRAT format.

    Uses line-by-line splitting by default for better NER performance.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        chunk_size: int = -1,
        filter_nested: bool = True,
    ):
        """Initialize CAS2 adapter with line-by-line splitting.

        Args:
            data_dir: Path to CAS2 data
            chunk_size: -1 for line-by-line (default), 0 for no chunking,
                        >0 for character-based chunks
            filter_nested: If True (default), keep only coarsest granularity
                          entities (matches CamemBERT-bio paper methodology)
        """
        super().__init__(
            data_dir=data_dir if data_dir else DEFAULT_CAS2_DIR,
            name="cas2",
            chunk_size=chunk_size,
            filter_nested=filter_nested,
        )

