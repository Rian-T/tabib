"""Dataset adapter abstractions."""

from tabib.data.base import DatasetAdapter
from tabib.data.agnews import AGNewsAdapter
from tabib.data.brat import BRATDatasetAdapter
from tabib.data.cas import CASAdapter, CAS1Adapter, CAS2Adapter
from tabib.data.cim10_mcqa import CIM10MCQAAdapter
from tabib.data.clister import CLISTERAdapter
from tabib.data.diamed import DiaMEDAdapter
from tabib.data.e3c import E3CAdapter
from tabib.data.essai import ESSAIAdapter
from tabib.data.emea import EMEAAdapter
from tabib.data.medline import MEDLINEAdapter
from tabib.data.fracco import (
    FRACCODocumentICDAdapter,
    FRACCOExpressionNERAdapter,
    FRACCOICDClassificationAdapter,
)
from tabib.data.frasimed import (
    FRASIMEDCANTEMISTAdapter,
    FRASIMEDDISTEMISTAdapter,
    FRASIMEDDocumentMultilabelAdapter,
    FRASIMEDMentionClassificationAdapter,
    FRASIMEDNERAdapter,
)
from tabib.data.french_med_mcqa_extended import FrenchMedMCQAExtendedAdapter
from tabib.data.jnlpba import JNLPBAAdapter
from tabib.data.mantragsc import MANTRAGSCAdapter
from tabib.data.meddialog import MedDialogWomenAdapter
from tabib.data.mediqal import (
    MediQAlMCQMAdapter,
    MediQAlMCQUAdapter,
    MediQAlOEQAdapter,
)
from tabib.data.mixed_mcqa import MixedMCQAAdapter
from tabib.data.morfitt import MORFITTAdapter
from tabib.data.quaero_emea_hf import QuaeroEMEAHFAdapter
from tabib.data.quaero_emea_token import QuaeroEMEATokenAdapter
from tabib.data.quaero_medline_hf import QuaeroMEDLINEHFAdapter
from tabib.data.quaero_medline_token import QuaeroMEDLINETokenAdapter
from tabib.data.e3c_token import E3CTokenAdapter
from tabib.data.cas1_token import CAS1TokenAdapter
from tabib.data.cas2_token import CAS2TokenAdapter
from tabib.data.sts import STSAdapter
from tabib.data.wikiann import WikiAnnAdapter

__all__ = [
    "DatasetAdapter",
    "WikiAnnAdapter",
    "AGNewsAdapter",
    "STSAdapter",
    "CLISTERAdapter",
    "DiaMEDAdapter",
    "ESSAIAdapter",
    "FRACCODocumentICDAdapter",
    "FRACCOICDClassificationAdapter",
    "FRACCOExpressionNERAdapter",
    "JNLPBAAdapter",
    "BRATDatasetAdapter",
    "EMEAAdapter",
    "MANTRAGSCAdapter",
    "MEDLINEAdapter",
    "MORFITTAdapter",
    "FrenchMedMCQAExtendedAdapter",
    "CASAdapter",
    "CAS1Adapter",
    "CAS2Adapter",
    "CIM10MCQAAdapter",
    "E3CAdapter",
    "MediQAlMCQUAdapter",
    "MediQAlMCQMAdapter",
    "MediQAlOEQAdapter",
    "MedDialogWomenAdapter",
    "MixedMCQAAdapter",
    "QuaeroEMEAHFAdapter",
    "QuaeroEMEATokenAdapter",
    "QuaeroMEDLINEHFAdapter",
    "QuaeroMEDLINETokenAdapter",
    "E3CTokenAdapter",
    "CAS1TokenAdapter",
    "CAS2TokenAdapter",
    "FRASIMEDNERAdapter",
    "FRASIMEDCANTEMISTAdapter",
    "FRASIMEDDISTEMISTAdapter",
    "FRASIMEDMentionClassificationAdapter",
    "FRASIMEDDocumentMultilabelAdapter",
]

