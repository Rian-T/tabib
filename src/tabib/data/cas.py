"""CAS dataset adapters."""

from pathlib import Path

from tabib.data.brat import BRATDatasetAdapter


DEFAULT_CAS_DIR = Path(__file__).resolve().parents[3] / "data" / "CAS"
DEFAULT_CAS1_DIR = Path(__file__).resolve().parents[3] / "data" / "CAS1"
DEFAULT_CAS2_DIR = Path(__file__).resolve().parents[3] / "data" / "CAS2"

# HuggingFace repos
HF_REPO_CAS = "rntc/tabib-cas"
HF_REPO_CAS1 = "rntc/tabib-cas1"
HF_REPO_CAS2 = "rntc/tabib-cas2"


class CASAdapter(BRATDatasetAdapter):
    """Adapter for the combined CAS dataset (CAS1 + CAS2) in BRAT format.

    Uses line-by-line splitting by default for better NER performance.

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-cas --local-dir $SCRATCH/tabib/data/rntc--tabib-cas --repo-type dataset
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        chunk_size: int = -1,
        filter_nested: bool = False,
    ):
        """Initialize combined CAS adapter with line-by-line splitting.

        Args:
            data_dir: Path to combined CAS data
            chunk_size: -1 for line-by-line (default), 0 for no chunking,
                        >0 for character-based chunks
            filter_nested: If False (default), keep all entities including nested ones.
                          Set to True for token-based NER (seqeval) which can't handle nesting.
        """
        super().__init__(
            data_dir=data_dir if data_dir else DEFAULT_CAS_DIR,
            name="cas",
            chunk_size=chunk_size,
            filter_nested=filter_nested,
            hf_repo=HF_REPO_CAS,
        )


class CAS1Adapter(BRATDatasetAdapter):
    """Adapter for the CAS1 dataset stored in BRAT format.

    Uses line-by-line splitting by default for better NER performance.

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-cas1 --local-dir $SCRATCH/tabib/data/rntc--tabib-cas1 --repo-type dataset
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        chunk_size: int = -1,
        filter_nested: bool = False,
    ):
        """Initialize CAS1 adapter with line-by-line splitting.

        Args:
            data_dir: Path to CAS1 data
            chunk_size: -1 for line-by-line (default), 0 for no chunking,
                        >0 for character-based chunks
            filter_nested: If False (default), keep all entities including nested ones.
                          Set to True for token-based NER (seqeval) which can't handle nesting.
        """
        super().__init__(
            data_dir=data_dir if data_dir else DEFAULT_CAS1_DIR,
            name="cas1",
            chunk_size=chunk_size,
            filter_nested=filter_nested,
            hf_repo=HF_REPO_CAS1,
        )


class CAS2Adapter(BRATDatasetAdapter):
    """Adapter for the CAS2 dataset stored in BRAT format.

    Uses line-by-line splitting by default for better NER performance.

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-cas2 --local-dir $SCRATCH/tabib/data/rntc--tabib-cas2 --repo-type dataset
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        chunk_size: int = -1,
        filter_nested: bool = False,
    ):
        """Initialize CAS2 adapter with line-by-line splitting.

        Args:
            data_dir: Path to CAS2 data
            chunk_size: -1 for line-by-line (default), 0 for no chunking,
                        >0 for character-based chunks
            filter_nested: If False (default), keep all entities including nested ones.
                          Set to True for token-based NER (seqeval) which can't handle nesting.
        """
        super().__init__(
            data_dir=data_dir if data_dir else DEFAULT_CAS2_DIR,
            name="cas2",
            chunk_size=chunk_size,
            filter_nested=filter_nested,
            hf_repo=HF_REPO_CAS2,
        )
