"""Tabib: Task-agnostic NLP evaluation framework."""

from tabib.config import RunConfig, TrainingConfig
from tabib.data.base import DatasetAdapter
from tabib.models.base import ModelAdapter
from tabib.registry import (
    get_dataset,
    get_model,
    get_task,
    register_dataset,
    register_model,
    register_task,
)
from tabib.tasks.base import Task

# Import and register components
from tabib import data, models, tasks  # noqa: F401

# Register all components
register_task("ner_token", tasks.NERTokenTask)
register_task("ner_span", tasks.NERSpanTask)
register_task("classification", tasks.ClassificationTask)
register_task("mcqa", tasks.MultipleChoiceTask)
register_task("similarity", tasks.SimilarityTask)
register_task("open_qa", tasks.OpenQATask)
register_dataset("wikiann_en", data.WikiAnnAdapter)
register_dataset("ag_news", data.AGNewsAdapter)
register_dataset("sts", data.STSAdapter)
register_dataset("fracco_icd_classification", data.FRACCOICDClassificationAdapter)
register_dataset("fracco_icd_top50", lambda: data.FRACCOICDClassificationAdapter(top_k=50))
register_dataset("fracco_icd_top200", lambda: data.FRACCOICDClassificationAdapter(top_k=200))
register_dataset("fracco_expression_ner", data.FRACCOExpressionNERAdapter)
register_dataset("jnlpba", data.JNLPBAAdapter)
register_dataset("emea", data.EMEAAdapter)
register_dataset("medline", data.MEDLINEAdapter)
register_dataset("french_med_mcqa_extended", data.FrenchMedMCQAExtendedAdapter)
register_dataset("cas1", data.CAS1Adapter)
register_dataset("cas2", data.CAS2Adapter)
register_dataset("e3c", data.E3CAdapter)
register_dataset("mediqal_mcqu", data.MediQAlMCQUAdapter)
register_dataset("mediqal_mcqm", data.MediQAlMCQMAdapter)
register_dataset("mediqal_oeq", data.MediQAlOEQAdapter)
register_dataset("mixed_mcqa", data.MixedMCQAAdapter)
register_dataset("clister", data.CLISTERAdapter)
register_dataset("diamed", data.DiaMEDAdapter)
register_dataset("essai", data.ESSAIAdapter)
register_dataset("mantragsc_medline", data.MANTRAGSCAdapter)
register_dataset("morfitt", data.MORFITTAdapter)
register_dataset("meddialog_women", data.MedDialogWomenAdapter)
register_dataset("cim10_mcqa", data.CIM10MCQAAdapter)
register_model("bert_token_ner", models.BERTTokenNERAdapter)
register_model("bert_text_cls", models.BERTTextClassificationAdapter)
register_model("bert_similarity", models.BERTSimilarityAdapter)
register_model("vllm_classification", models.VLLMClassificationAdapter)
register_model("vllm_ner", models.VLLMNERAdapter)
register_model("vllm_open_qa", models.VLLMOpenQAAdapter)
register_model("gliner_zero_shot", models.GLiNERZeroShotNERAdapter)
register_model("lora_sft", models.LoRASFTAdapter)

__all__ = [
    "Task",
    "DatasetAdapter",
    "ModelAdapter",
    "RunConfig",
    "TrainingConfig",
    "register_task",
    "register_dataset",
    "register_model",
    "get_task",
    "get_dataset",
    "get_model",
]

