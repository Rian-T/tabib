#!/usr/bin/env python
"""Upload local tabib datasets to HuggingFace Hub.

Usage:
    python scripts/upload_datasets_to_hf.py --dataset cas1
    python scripts/upload_datasets_to_hf.py --all
    python scripts/upload_datasets_to_hf.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import Dataset

NAMESPACE = "rntc"
PREFIX = "tabib"


def load_brat_dataset(adapter_class, **kwargs) -> dict[str, Dataset]:
    """Load a BRAT-format dataset using its adapter."""
    adapter = adapter_class(**kwargs)
    return adapter.load_splits()


def upload_dataset(name: str, splits: dict[str, Dataset], private: bool = False) -> None:
    """Upload dataset splits to HuggingFace Hub."""
    repo_id = f"{NAMESPACE}/{PREFIX}-{name}"

    print(f"Uploading {name} to {repo_id}...")

    # Combine splits and push
    from datasets import DatasetDict
    ds_dict = DatasetDict(splits)
    ds_dict.push_to_hub(repo_id, private=private)

    print(f"  Uploaded: {repo_id}")
    for split_name, ds in splits.items():
        print(f"    {split_name}: {len(ds)} samples")


def upload_cas1(private: bool = False) -> None:
    """Upload CAS1 NER dataset."""
    from tabib.data.cas import CAS1Adapter
    # Use no chunking to keep original documents
    splits = load_brat_dataset(CAS1Adapter, chunk_size=0, filter_nested=False)
    upload_dataset("cas1", splits, private)


def upload_cas2(private: bool = False) -> None:
    """Upload CAS2 NER dataset."""
    from tabib.data.cas import CAS2Adapter
    splits = load_brat_dataset(CAS2Adapter, chunk_size=0, filter_nested=False)
    upload_dataset("cas2", splits, private)


def upload_emea(private: bool = False) -> None:
    """Upload EMEA NER dataset."""
    from tabib.data.emea import EMEAAdapter
    splits = load_brat_dataset(EMEAAdapter, chunk_size=0, filter_nested=False)
    upload_dataset("emea", splits, private)


def upload_medline(private: bool = False) -> None:
    """Upload MEDLINE NER dataset."""
    from tabib.data.medline import MEDLINEAdapter
    splits = load_brat_dataset(MEDLINEAdapter)
    upload_dataset("medline", splits, private)


def upload_mantragsc(private: bool = False) -> None:
    """Upload MANTRAGSC NER dataset (all sources combined)."""
    from tabib.data.mantragsc import MANTRAGSCAdapter

    # Load all three sources
    all_records = {"train": [], "val": [], "test": []}

    for source in ["medline", "emea", "patent"]:
        adapter = MANTRAGSCAdapter(source=source)
        splits = adapter.load_splits()
        for split_name, ds in splits.items():
            if split_name in all_records:
                # Add source field to each record
                for record in ds:
                    record_with_source = dict(record)
                    record_with_source["source"] = source
                    all_records[split_name].append(record_with_source)

    combined = {
        split: Dataset.from_list(records)
        for split, records in all_records.items()
        if records
    }
    upload_dataset("mantragsc", combined, private)


def upload_fracco(private: bool = False) -> None:
    """Upload FRACCO dataset (ICD classification + NER)."""
    from tabib.data.fracco import FRACCOICDClassificationAdapter, FRACCOExpressionNERAdapter

    # ICD Classification (no filtering for full dataset)
    icd_adapter = FRACCOICDClassificationAdapter(min_samples=1, top_k=None)
    icd_splits = icd_adapter.load_splits()
    upload_dataset("fracco-icd", icd_splits, private)

    # Expression NER
    ner_adapter = FRACCOExpressionNERAdapter()
    ner_splits = ner_adapter.load_splits()
    upload_dataset("fracco-ner", ner_splits, private)


def upload_meddialog(private: bool = False) -> None:
    """Upload MedDialog-FR Women dataset."""
    from tabib.data.meddialog import MedDialogWomenAdapter

    adapter = MedDialogWomenAdapter()
    splits = adapter.load_splits()
    upload_dataset("meddialog-fr", splits, private)


def upload_frasimed_cantemist(private: bool = False) -> None:
    """Upload FRASIMED-CANTEMIST NER dataset."""
    from tabib.data.frasimed import FRASIMEDCANTEMISTAdapter

    adapter = FRASIMEDCANTEMISTAdapter()
    splits = adapter.load_splits()
    upload_dataset("frasimed-cantemist", splits, private)


def upload_frasimed_distemist(private: bool = False) -> None:
    """Upload FRASIMED-DISTEMIST NER dataset."""
    from tabib.data.frasimed import FRASIMEDDISTEMISTAdapter

    adapter = FRASIMEDDISTEMISTAdapter()
    splits = adapter.load_splits()
    upload_dataset("frasimed-distemist", splits, private)


def upload_frasimed_cantemist_doc(private: bool = False) -> None:
    """Upload FRASIMED-CANTEMIST document-level multilabel dataset."""
    from tabib.data.frasimed import FRASIMEDDocumentMultilabelAdapter

    adapter = FRASIMEDDocumentMultilabelAdapter(subset="cantemist", top_k=100)
    splits = adapter.load_splits()
    upload_dataset("frasimed-cantemist-doc", splits, private)


def upload_frasimed_distemist_doc(private: bool = False) -> None:
    """Upload FRASIMED-DISTEMIST document-level multilabel dataset."""
    from tabib.data.frasimed import FRASIMEDDocumentMultilabelAdapter

    adapter = FRASIMEDDocumentMultilabelAdapter(subset="distemist", top_k=100)
    splits = adapter.load_splits()
    upload_dataset("frasimed-distemist-doc", splits, private)


DATASETS = {
    "cas1": upload_cas1,
    "cas2": upload_cas2,
    "emea": upload_emea,
    "medline": upload_medline,
    "mantragsc": upload_mantragsc,
    "fracco": upload_fracco,
    "meddialog": upload_meddialog,
    "frasimed_cantemist": upload_frasimed_cantemist,
    "frasimed_distemist": upload_frasimed_distemist,
    "frasimed_cantemist_doc": upload_frasimed_cantemist_doc,
    "frasimed_distemist_doc": upload_frasimed_distemist_doc,
}


def main():
    parser = argparse.ArgumentParser(description="Upload tabib datasets to HuggingFace Hub")
    parser.add_argument("--dataset", "-d", choices=list(DATASETS.keys()), help="Dataset to upload")
    parser.add_argument("--all", "-a", action="store_true", help="Upload all datasets")
    parser.add_argument("--list", "-l", action="store_true", help="List available datasets")
    parser.add_argument("--private", "-p", action="store_true", help="Make datasets private")

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name in DATASETS:
            print(f"  - {name} -> {NAMESPACE}/{PREFIX}-{name}")
        return

    if args.all:
        for name, upload_fn in DATASETS.items():
            try:
                upload_fn(private=args.private)
            except Exception as e:
                print(f"  Error uploading {name}: {e}")
        return

    if args.dataset:
        DATASETS[args.dataset](private=args.private)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
