"""vLLM-based NER model adapter with XML-tagged entity extraction."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import weave

from tabib.models.base import ModelAdapter
from tabib.models.vllm_common import (
    VLLMEngine,
    build_messages,
    chat_with_vllm,
    create_vllm_engine,
)
from tabib.tasks.ner_span import NERSpanTask


@dataclass
class _VLLMResources:
    engine: VLLMEngine
    entity_types: list[str]
    system_prompt: str
    user_prompt_template: str
    enable_thinking: bool | None
    chat_template_kwargs: dict[str, Any]
    use_chat: bool
    few_shot_examples: list[dict[str, str]]  # List of {input, output} example pairs


class VLLMNERAdapter(ModelAdapter):
    """NER adapter that runs inference with vLLM using context-based extraction."""

    def __init__(self) -> None:
        self._resources: _VLLMResources | None = None
        self._default_system_prompt = (
            "You are a biomedical text analyzer specialized in named entity recognition. "
            "Tag all entities in the text using XML-style tags. "
            "Output the EXACT same text with entities wrapped in tags like <ENTITY_TYPE>text</ENTITY_TYPE>. "
            "Nested entities should use nested tags. "
            "Do not modify the text in any way, only add tags around entities."
        )
        # Entity type definitions for biomedical NER
        # MEDLINE/EMEA entity types
        self._entity_definitions = {
            "ANAT": "Anatomy (body parts, organs, cells)",
            "CHEM": "Chemical and Drugs (medications, compounds)",
            "DEVI": "Devices (medical equipment, instruments)",
            "DISO": "Disorders (diseases, conditions, symptoms)",
            "GEOG": "Geographic Areas (locations, regions)",
            "LIVB": "Living Beings (organisms, species)",
            "OBJC": "Objects (physical objects)",
            "PHEN": "Phenomena (biological phenomena)",
            "PHYS": "Physiology (physiological processes)",
            "PROC": "Procedures (medical procedures, treatments)",
            # CAS1 entity types
            "sosy": "Signs and Symptoms (clinical signs, symptoms)",
            "pathologie": "Pathology (diseases, medical conditions)",
            # CAS2 entity types
            "moment": "Temporal (time references, duration, moments)",
            "mode": "Administration Mode (route, method of administration)",
            "substance": "Substance (drugs, medications, chemical compounds)",
            "anatomie": "Anatomy (body parts, organs, anatomical structures)",
            "examen": "Examination (clinical tests, imaging, diagnostic procedures)",
            "traitement": "Treatment (therapeutic interventions, procedures)",
            "valeur": "Value (numerical measurements, lab values, quantities)",
            "dose": "Dose (medication dosages, amounts)",
        }
        # User prompt template - {few_shot} and {text} will be filled in _build_prompt
        # V2: Clear section headers to prevent few-shot leakage
        self._default_user_prompt = (
            "=== TASK: French Biomedical Named Entity Recognition ===\n\n"
            "Tag entities in text using XML format: <TYPE>entity</TYPE>\n\n"
            "Entity types:\n{entity_definitions}\n"
            "{few_shot}\n"
            "=== YOUR INPUT (annotate ONLY this text, NOT the examples above) ===\n\n"
            "Input: \"{text}\"\n"
            "Output:"
        )

    @property
    def name(self) -> str:
        return "vllm_ner"

    @property
    def supports_finetune(self) -> bool:
        return False

    @weave.op()
    def _log_llm_call(self, input_text: str, prompt: str, output: str) -> dict:
        """Log LLM call details to Weave for debugging."""
        return {
            "input_text": input_text,
            "prompt": prompt,
            "raw_output": output,
        }

    def build_model(
        self,
        task: Any,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        enable_thinking: bool | None = True,
        use_chat: bool = True,
        num_few_shot: int = 3,
        train_data: Any | None = None,
        sampling_overrides: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _VLLMResources:
        """Build vLLM model with structured outputs for NER.
        
        Args:
            task: NERSpanTask instance
            model_name_or_path: Model name or path
            system_prompt: Optional custom system prompt
            user_prompt_template: Optional custom user prompt template
            enable_thinking: Toggle for the vLLM chat template.
            sampling_overrides: Overrides for the shared sampling defaults.
            chat_template_kwargs: Extra kwargs forwarded to ``llm.chat``'s template.
            **kwargs: Additional vLLM engine arguments
            
        Returns:
            VLLMResources with LLM engine and configuration
        """
        if not isinstance(task, NERSpanTask):
            raise ValueError(f"Expected NERSpanTask, got {type(task)}")

        entity_types = task.label_space
        if not entity_types:
            raise ValueError("NERSpanTask must provide entity types (label_space)")

        # Configure sampling without structured outputs (plain text with XML tags)
        sampling_kwargs: dict[str, Any] = dict(sampling_overrides or {})

        temperature = kwargs.pop("temperature", None)
        if "temperature" not in sampling_kwargs:
            sampling_kwargs["temperature"] = 0.0 if temperature is None else temperature

        max_tokens = kwargs.pop("max_tokens", None)
        if "max_tokens" not in sampling_kwargs:
            sampling_kwargs["max_tokens"] = 1024 if max_tokens is None else max_tokens

        # Add stop tokens to prevent model from generating additional examples
        if "stop" not in sampling_kwargs:
            sampling_kwargs["stop"] = ["\n\n===", "\n\nInput:", "\n\nExample", "\n\nText:"]

        engine = create_vllm_engine(
            model_name_or_path,
            sampling_overrides=sampling_kwargs,
            **kwargs,
        )

        # Create few-shot examples from train data
        few_shot_examples = []
        if train_data and num_few_shot > 0:
            few_shot_examples = self._create_few_shot_examples(
                train_data, entity_types, num_few_shot
            )

        resources = _VLLMResources(
            engine=engine,
            entity_types=entity_types,
            system_prompt=system_prompt or self._default_system_prompt,
            user_prompt_template=user_prompt_template or self._default_user_prompt,
            enable_thinking=enable_thinking,
            chat_template_kwargs=dict(chat_template_kwargs or {}),
            use_chat=use_chat,
            few_shot_examples=few_shot_examples,
        )
        self._resources = resources
        return resources

    def _create_few_shot_examples(
        self, train_data: Any, entity_types: list[str], num_examples: int
    ) -> list[dict[str, str]]:
        """Create few-shot examples from training data.

        Args:
            train_data: Training dataset
            entity_types: List of entity types
            num_examples: Number of examples to create

        Returns:
            List of {input, output} dictionaries
        """
        examples = []

        # Sample examples from train data (random each time, no fixed seed)
        import random

        # Convert train_data to list if needed
        train_list = list(train_data) if hasattr(train_data, '__iter__') else []

        # Filter to only examples WITH entities
        train_with_entities = [
            ex for ex in train_list
            if (ex.get('spans') or ex.get('entities', []))
        ]

        if len(train_with_entities) < num_examples:
            num_examples = len(train_with_entities)

        if num_examples == 0:
            return []

        sampled = random.sample(train_with_entities, num_examples)

        for example in sampled:
            text = example.get('text', '')
            # Try both 'spans' and 'entities' keys
            spans = example.get('spans') or example.get('entities', [])

            if not text or not spans:
                continue

            # Create XML-tagged output
            tagged_text = self._create_tagged_text(text, spans)

            examples.append({
                'input': text,
                'output': tagged_text
            })

        return examples

    def _create_tagged_text(self, text: str, spans: list[dict]) -> str:
        """Create XML-tagged version of text from spans.

        Handles nested entities by inserting opening/closing tags at the right positions.
        Example: "potentiel visuel évoqué" with nested "visuel" becomes:
        <PHYS>potentiel <PHYS>visuel</PHYS> évoqué</PHYS>

        Args:
            text: Original text
            spans: List of entity spans with start, end, label

        Returns:
            Text with XML tags around entities (including nested)
        """
        if not spans:
            return text

        # Create list of tag events (position, is_open, label, span_length)
        # span_length is used for sorting: longer spans open first, close last
        events = []
        for span in spans:
            start = span['start']
            end = span['end']
            label = span['label']
            length = end - start
            # Opening tag: sort by position, then longer spans first (negative length)
            events.append((start, 0, -length, label, 'open'))
            # Closing tag: sort by position, then shorter spans first (positive length)
            events.append((end, 1, length, label, 'close'))

        # Sort events: by position, then opens before closes at same position,
        # then by length (longer opens first, shorter closes first)
        events.sort(key=lambda x: (x[0], x[1], x[2]))

        # Build result
        result = []
        last_pos = 0

        for pos, _, _, label, tag_type in events:
            # Add text before this position
            if pos > last_pos:
                result.append(text[last_pos:pos])
                last_pos = pos

            # Add tag
            if tag_type == 'open':
                result.append(f"<{label}>")
            else:
                result.append(f"</{label}>")

        # Add remaining text after last tag
        if last_pos < len(text):
            result.append(text[last_pos:])

        return ''.join(result)

    def get_trainer(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("vLLM NER does not support fine-tuning")

    def predict(self, model: Any, inputs: Any, **kwargs: Any) -> list[dict]:
        """Run NER inference and return document-level predictions.

        Uses batched inference for efficiency - all prompts are sent to vLLM together.

        Args:
            model: VLLMResources instance
            inputs: Dataset with chunked documents
            **kwargs: Additional arguments (batch_size, etc.)

        Returns:
            List of documents with predicted entities at document level
        """
        resources = model if isinstance(model, _VLLMResources) else self._resources
        if resources is None:
            raise ValueError("Model not built. Call build_model first.")

        enable_thinking = kwargs.pop("enable_thinking", resources.enable_thinking)
        chat_template_kwargs = dict(resources.chat_template_kwargs)
        chat_template_kwargs.update(kwargs.pop("chat_template_kwargs", {}))
        chat_call_kwargs = dict(kwargs)

        # Collect all chunk metadata and build all prompts
        chunk_metadata = []
        all_prompts = []
        all_messages = []

        for example in inputs:
            doc_id = example['doc_id']
            chunk_id = example.get('chunk_id', 0)
            chunk_offset = example.get('chunk_offset', 0)
            text = example['text']

            chunk_metadata.append({
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'chunk_offset': chunk_offset,
                'text': text,
            })

            # Build prompt for this chunk
            prompt = self._build_prompt(text, resources)
            all_prompts.append(prompt)

            if resources.use_chat:
                messages = build_messages(prompt, system_prompt=resources.system_prompt)
                all_messages.append(messages)

        # Run batched inference
        if resources.use_chat:
            outputs = chat_with_vllm(
                resources.engine,
                all_messages,
                enable_thinking=enable_thinking,
                chat_template_kwargs=chat_template_kwargs,
                **chat_call_kwargs,
            )
        else:
            # Completion mode: prepend system prompt to each prompt
            full_prompts = []
            for prompt in all_prompts:
                if resources.system_prompt:
                    full_prompts.append(f"{resources.system_prompt}\n\n{prompt}")
                else:
                    full_prompts.append(prompt)
            outputs = resources.engine.llm.generate(
                full_prompts,
                sampling_params=resources.engine.sampling_params,
                **chat_call_kwargs,
            )

        # Parse outputs and build chunk predictions
        chunk_predictions = []
        for i, (meta, output) in enumerate(zip(chunk_metadata, outputs)):
            if not output.outputs:
                entities = []
            else:
                generated_text = output.outputs[0].text.strip()
                # Log raw LLM output
                self._log_llm_call(meta['text'], all_prompts[i], generated_text)
                # Parse XML-tagged output
                entities = self._parse_xml_tags(generated_text, meta['text'])

            chunk_predictions.append({
                'doc_id': meta['doc_id'],
                'chunk_id': meta['chunk_id'],
                'chunk_offset': meta['chunk_offset'],
                'spans': entities,
            })

        # Reassemble chunks into documents
        documents = self._reassemble_chunks(chunk_predictions)

        return documents

    def _build_prompt(self, text: str, resources: _VLLMResources) -> str:
        """Build the prompt for a single chunk.

        Args:
            text: Chunk text
            resources: VLLMResources with model and config

        Returns:
            Formatted prompt string
        """
        # Build few-shot examples string with clear section header
        few_shot_str = ""
        if resources.few_shot_examples:
            few_shot_str = "\n=== EXAMPLES (for format reference only, DO NOT include in output) ===\n"
            for i, example in enumerate(resources.few_shot_examples, 1):
                few_shot_str += f"\nExample {i}:\nInput: \"{example['input'].strip()}\"\nOutput: \"{example['output'].strip()}\"\n"

        # Build entity definitions string
        entity_defs = "\n".join([
            f"- {etype}: {self._entity_definitions.get(etype, etype)}"
            for etype in resources.entity_types
        ])

        # Format prompt - text goes at the end
        return resources.user_prompt_template.format(
            entity_definitions=entity_defs,
            few_shot=few_shot_str,
            text=text
        )

    @weave.op()
    def _predict_chunk(
        self,
        text: str,
        resources: _VLLMResources,
        *,
        enable_thinking: bool | None,
        chat_template_kwargs: dict[str, Any],
        chat_call_kwargs: dict[str, Any],
    ) -> list[dict]:
        """Predict entities for a single chunk.

        Args:
            text: Chunk text
            resources: VLLMResources with model and config

        Returns:
            List of entity spans with character offsets
        """
        # Build few-shot examples string with clear section header
        few_shot_str = ""
        if resources.few_shot_examples:
            few_shot_str = "\n=== EXAMPLES (for format reference only, DO NOT include in output) ===\n"
            for i, example in enumerate(resources.few_shot_examples, 1):
                few_shot_str += f"\nExample {i}:\nInput: \"{example['input'].strip()}\"\nOutput: \"{example['output'].strip()}\"\n"

        # Build entity definitions string
        entity_defs = "\n".join([
            f"- {etype}: {self._entity_definitions.get(etype, etype)}"
            for etype in resources.entity_types
        ])

        # Format prompt - text goes at the end
        user_prompt = resources.user_prompt_template.format(
            entity_definitions=entity_defs,
            few_shot=few_shot_str,
            text=text
        )

        # Generate with vLLM
        if resources.use_chat:
            # Chat mode: use chat API with messages
            messages = build_messages(user_prompt, system_prompt=resources.system_prompt)
            outputs = chat_with_vllm(
                resources.engine,
                [messages],
                enable_thinking=enable_thinking,
                chat_template_kwargs=chat_template_kwargs,
                **chat_call_kwargs,
            )
        else:
            # Completion mode: prepend system prompt to user prompt
            if resources.system_prompt:
                full_prompt = f"{resources.system_prompt}\n\n{user_prompt}"
            else:
                full_prompt = user_prompt
            outputs = resources.engine.llm.generate(
                [full_prompt],
                sampling_params=resources.engine.sampling_params,
                **chat_call_kwargs,
            )
        
        # Parse output
        if not outputs or not outputs[0].outputs:
            return []

        generated_text = outputs[0].outputs[0].text.strip()

        # Log raw LLM output (visible in Weave trace as return value)
        self._log_llm_call(text, user_prompt, generated_text)

        # Parse XML-tagged output
        spans = self._parse_xml_tags(generated_text, text)
        return spans

    def _normalize_text(self, text: str) -> str:
        """Normalize text for fuzzy matching.

        Handles common variations:
        - œ <-> oe
        - æ <-> ae
        - Accents (é -> e, à -> a, etc.)
        - Multiple spaces -> single space
        - Case insensitive
        """
        import unicodedata

        # Replace ligatures
        text = text.replace('œ', 'oe').replace('Œ', 'OE')
        text = text.replace('æ', 'ae').replace('Æ', 'AE')

        # Remove accents
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

        # Normalize whitespace and lowercase
        text = ' '.join(text.split()).lower()

        return text

    def _fuzzy_find_entity(self, entity_text: str, search_text: str, search_offset: int = 0) -> tuple[int, int] | None:
        """Find entity in search_text using fuzzy matching.

        Args:
            entity_text: Entity to find
            search_text: Text to search in
            search_offset: Offset in original text where search_text starts

        Returns:
            Tuple of (start, end) in original coordinates, or None
        """
        # Try exact match first
        idx = search_text.find(entity_text)
        if idx != -1:
            return (search_offset + idx, search_offset + idx + len(entity_text))

        # Try normalized fuzzy match
        norm_entity = self._normalize_text(entity_text)
        norm_search = self._normalize_text(search_text)

        idx = norm_search.find(norm_entity)
        if idx != -1:
            # Find the original boundaries by counting characters
            # Map normalized position back to original position
            orig_idx = 0
            norm_idx = 0

            # Find start position
            while norm_idx < idx and orig_idx < len(search_text):
                if self._normalize_text(search_text[orig_idx]):
                    norm_idx += len(self._normalize_text(search_text[orig_idx]))
                orig_idx += 1

            start = orig_idx

            # Find end position (match length in normalized space)
            norm_end = idx + len(norm_entity)
            while norm_idx < norm_end and orig_idx < len(search_text):
                if self._normalize_text(search_text[orig_idx]):
                    norm_idx += len(self._normalize_text(search_text[orig_idx]))
                orig_idx += 1

            end = orig_idx

            return (search_offset + start, search_offset + end)

        return None

    def _parse_xml_tags(self, tagged_text: str, original_text: str) -> list[dict]:
        """Parse XML-tagged text and extract entity spans.

        Args:
            tagged_text: Model output with XML tags
            original_text: Original input text without tags

        Returns:
            List of entity spans with start, end, label, text
        """
        spans = []
        import re

        # Find all opening tags with their positions
        # Match both uppercase (ANAT, CHEM) and lowercase (sosy, pathologie) entity types
        tag_pattern = r'<([A-Za-z_]+)>([^<]+?)</\1>'

        for match in re.finditer(tag_pattern, tagged_text):
            label = match.group(1)
            entity_text = match.group(2).strip()

            if not entity_text:
                continue

            # Find this entity in the original text starting from where we left off
            # Calculate position offset by removing all tags before this match
            text_before_match = tagged_text[:match.start()]
            text_without_tags = re.sub(r'<[^>]+>', '', text_before_match)

            # Find entity in original text around this position
            search_start = len(text_without_tags)
            search_window = original_text[max(0, search_start - 50):search_start + len(entity_text) + 100]
            search_offset = max(0, search_start - 50)

            # Try fuzzy find
            result = self._fuzzy_find_entity(entity_text, search_window, search_offset)

            if result:
                abs_start, abs_end = result
                spans.append({
                    'start': abs_start,
                    'end': abs_end,
                    'label': label,
                    'text': original_text[abs_start:abs_end],
                })
            else:
                print(f"Warning: Could not locate entity '{entity_text}' (label: {label}) in original text")

        return spans

    def _locate_entity_by_context(self, text: str, entity_data: dict) -> dict | None:
        """Locate entity in text using context for disambiguation.
        
        Args:
            text: Source text
            entity_data: Dictionary with 'label', 'text', 'context_before', 'context_after'
            
        Returns:
            Entity span dict with 'start', 'end', 'label', 'text' or None if not found
        """
        label = entity_data.get('label', '')
        entity_text = entity_data.get('text', '')
        context_before = entity_data.get('context_before', '')
        context_after = entity_data.get('context_after', '')
        
        if not entity_text:
            return None
        
        # Build search patterns with varying context
        # Try full context first, then fallback to partial context, then entity only
        patterns_to_try = []
        
        # Full context
        if context_before and context_after:
            patterns_to_try.append((context_before, entity_text, context_after))
        
        # Only context before
        if context_before:
            patterns_to_try.append((context_before, entity_text, ''))
        
        # Only context after
        if context_after:
            patterns_to_try.append(('', entity_text, context_after))
        
        # Entity text only (last resort)
        patterns_to_try.append(('', entity_text, ''))
        
        # Try each pattern
        for ctx_before, ent_txt, ctx_after in patterns_to_try:
            result = self._search_with_pattern(text, ctx_before, ent_txt, ctx_after)
            if result is not None:
                start, end = result
                return {
                    'start': start,
                    'end': end,
                    'label': label,
                    'text': text[start:end],
                }
        
        # Entity not found
        print(f"Warning: Could not locate entity '{entity_text}' (label: {label}) in text")
        return None

    def _search_with_pattern(
        self, text: str, context_before: str, entity_text: str, context_after: str
    ) -> tuple[int, int] | None:
        """Search for entity using context pattern.
        
        Args:
            text: Source text
            context_before: Context before entity
            entity_text: Entity text
            context_after: Context after entity
            
        Returns:
            Tuple of (start, end) character offsets for entity, or None if not found
        """
        # Build search pattern
        # Use word boundaries for context to avoid partial matches
        pattern_parts = []
        
        if context_before:
            # Escape special regex chars and allow flexible whitespace
            escaped_before = re.escape(context_before.strip())
            pattern_parts.append(escaped_before + r'\s+')
        
        # Capture the entity text
        escaped_entity = re.escape(entity_text)
        pattern_parts.append(f'({escaped_entity})')
        
        if context_after:
            escaped_after = re.escape(context_after.strip())
            pattern_parts.append(r'\s+' + escaped_after)
        
        pattern = ''.join(pattern_parts)
        
        # Search for pattern
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return the entity portion (group 1)
                return match.start(1), match.end(1)
        except re.error:
            # Regex error, try simpler search
            pass
        
        # Fallback: simple substring search for entity text only
        if not context_before and not context_after:
            idx = text.find(entity_text)
            if idx != -1:
                return idx, idx + len(entity_text)
        
        return None

    def _reassemble_chunks(self, chunk_predictions: list[dict]) -> list[dict]:
        """Reassemble chunk predictions into document-level predictions.
        
        Args:
            chunk_predictions: List of dicts with doc_id, chunk_id, chunk_offset, spans
            
        Returns:
            List of documents with entities at document level
        """
        # Group by document
        doc_groups = defaultdict(list)
        for chunk_pred in chunk_predictions:
            doc_groups[chunk_pred['doc_id']].append(chunk_pred)
        
        # Reassemble each document
        documents = []
        for doc_id, chunks in doc_groups.items():
            # Sort chunks by chunk_id
            chunks.sort(key=lambda x: x['chunk_id'])
            
            # Merge spans, adjusting offsets to document level
            all_spans = []
            for chunk in chunks:
                chunk_offset = chunk['chunk_offset']
                for span in chunk['spans']:
                    # Adjust to document-level offsets
                    all_spans.append({
                        'start': span['start'] + chunk_offset,
                        'end': span['end'] + chunk_offset,
                        'label': span['label'],
                        'text': span['text'],
                    })
            
            documents.append({
                'doc_id': doc_id,
                'entities': all_spans,
            })
        
        return documents

