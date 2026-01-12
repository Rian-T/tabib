"""EMEA dataset adapter."""

from pathlib import Path

from tabib.data.brat import BRATDatasetAdapter


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "EMEA"
HF_REPO = "rntc/tabib-emea"


class EMEAAdapter(BRATDatasetAdapter):
    """Adapter for the EMEA dataset in BRAT format.

    Uses line-by-line splitting by default since EMEA documents are very long
    (25K+ chars) but there are only 11 train / 15 test documents.
    Each line becomes one sample for better NER performance.

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-emea --local-dir $SCRATCH/tabib/data/rntc--tabib-emea --repo-type dataset
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        chunk_size: int = -1,
        filter_nested: bool = False,
    ):
        """Initialize EMEA adapter with line-by-line splitting.

        Args:
            data_dir: Path to EMEA data
            chunk_size: -1 for line-by-line (default), 0 for no chunking,
                        >0 for character-based chunks
            filter_nested: If False (default), keep all entities including nested ones.
                          Set to True for token-based NER (seqeval) which can't handle nesting.
        """
        super().__init__(
            data_dir=data_dir if data_dir else DEFAULT_DATA_DIR,
            name="emea",
            chunk_size=chunk_size,
            filter_nested=filter_nested,
            hf_repo=HF_REPO,
        )

