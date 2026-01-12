"""Generic BRAT format dataset adapter."""

import re
from pathlib import Path
from typing import Any

from datasets import Dataset

from tabib.data.base import DatasetAdapter


class BRATDatasetAdapter(DatasetAdapter):
    """Generic adapter for BRAT format datasets.

    Automatically detects entity types by scanning annotations.
    Supports nested entities.
    Supports chunking long documents at newline boundaries.
    """

    def __init__(
        self,
        data_dir: str | Path,
        name: str = "brat",
        chunk_size: int = 0,
        filter_nested: bool = False,
    ):
        """Initialize BRAT adapter.

        Args:
            data_dir: Path to BRAT data directory
            name: Dataset name
            chunk_size: Max characters per chunk.
                        0 = no chunking (default)
                        -1 = split line by line (one line = one sample)
                        >0 = group lines up to chunk_size chars
            filter_nested: If True, keep only coarsest granularity entities
                          (remove nested entities, keeping outermost spans).
                          This matches CamemBERT-bio paper methodology.
        """
        self.data_dir = Path(data_dir)
        self._name = name
        self._entity_types: set[str] = set()
        self.chunk_size = chunk_size
        self.filter_nested = filter_nested
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def entity_types(self) -> list[str]:
        """Return sorted list of discovered entity types."""
        return sorted(self._entity_types)
    
    def load_splits(self) -> dict[str, Dataset]:
        """Load train/dev/test splits from BRAT format."""
        splits = {}
        
        for split_name in ['train', 'dev', 'test']:
            split_dir = self.data_dir / split_name
            if not split_dir.exists():
                continue
            
            documents = self._load_split_documents(split_dir)
            if documents:
                splits[split_name] = Dataset.from_list(documents)
        
        return splits
    
    def _load_split_documents(self, split_dir: Path) -> list[dict[str, Any]]:
        """Load all documents from a split directory."""
        documents = []
        txt_files = sorted(split_dir.glob('*.txt'))

        for txt_path in txt_files:
            ann_path = txt_path.with_suffix('.ann')
            if not ann_path.exists():
                continue

            doc = self._load_document(txt_path, ann_path)

            # Line-by-line splitting (-1)
            if self.chunk_size == -1:
                chunks = self._split_by_lines(doc)
                documents.extend(chunks)
            # Chunk if enabled and document is large
            elif self.chunk_size > 0 and len(doc['text']) > self.chunk_size:
                chunks = self._chunk_document(doc)
                documents.extend(chunks)
            else:
                documents.append(doc)

        return documents

    def _chunk_document(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Split a document into chunks at newline boundaries.

        Preserves entity annotations by adjusting offsets.
        Only entities fully within a chunk are included.
        """
        text = doc['text']
        entities = doc['entities']
        doc_id = doc['doc_id']

        # Find all newline positions for splitting
        newline_positions = [0]
        for match in re.finditer(r'\n', text):
            newline_positions.append(match.end())
        newline_positions.append(len(text))

        chunks = []
        chunk_start = 0
        chunk_idx = 0

        while chunk_start < len(text):
            # Find best split point near chunk_size
            chunk_end = min(chunk_start + self.chunk_size, len(text))

            # If not at end, find nearest newline before chunk_end
            if chunk_end < len(text):
                best_split = chunk_start
                for pos in newline_positions:
                    if pos <= chunk_end and pos > best_split:
                        best_split = pos
                # If no good split found, force split at chunk_size
                if best_split == chunk_start:
                    chunk_end = chunk_start + self.chunk_size
                else:
                    chunk_end = best_split

            chunk_text = text[chunk_start:chunk_end]

            # Get entities within this chunk (fully contained)
            chunk_entities = []
            for ent in entities:
                if ent['start'] >= chunk_start and ent['end'] <= chunk_end:
                    chunk_entities.append({
                        'start': ent['start'] - chunk_start,
                        'end': ent['end'] - chunk_start,
                        'label': ent['label'],
                        'text': ent['text']
                    })

            chunks.append({
                'doc_id': f"{doc_id}_chunk{chunk_idx}",
                'text': chunk_text,
                'entities': chunk_entities
            })

            chunk_start = chunk_end
            chunk_idx += 1

        return chunks

    def _split_by_lines(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Split a document into one sample per line.

        Each non-empty line becomes a separate sample.
        Entities are only included if fully contained within a line.
        """
        text = doc['text']
        entities = doc['entities']
        doc_id = doc['doc_id']

        lines = []
        line_start = 0

        for i, char in enumerate(text):
            if char == '\n':
                line_text = text[line_start:i]
                lines.append((line_start, i, line_text))
                line_start = i + 1

        # Handle last line (no trailing newline)
        if line_start < len(text):
            line_text = text[line_start:]
            lines.append((line_start, len(text), line_text))

        samples = []
        for line_idx, (start, end, line_text) in enumerate(lines):
            # Skip empty lines
            if not line_text.strip():
                continue

            # Get entities fully within this line
            line_entities = []
            for ent in entities:
                if ent['start'] >= start and ent['end'] <= end:
                    line_entities.append({
                        'start': ent['start'] - start,
                        'end': ent['end'] - start,
                        'label': ent['label'],
                        'text': ent['text']
                    })

            samples.append({
                'doc_id': f"{doc_id}_line{line_idx}",
                'text': line_text,
                'entities': line_entities
            })

        return samples

    def _load_document(self, txt_path: Path, ann_path: Path) -> dict[str, Any]:
        """Load a single BRAT document."""
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Pass text for fragment merging (BRAT auto-splits at line boundaries)
        entities = self._parse_annotations(ann_path, text=text)

        # Filter nested entities if requested (keep coarsest granularity)
        if self.filter_nested:
            entities = self._filter_nested_entities(entities)

        return {
            'doc_id': txt_path.stem,
            'text': text,
            'entities': entities
        }

    def _filter_nested_entities(
        self, entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Keep only coarsest granularity entities (remove nested ones).

        This matches CamemBERT-bio paper methodology: "kept only the entities
        with the coarsest granularity" for QUAERO datasets.

        For overlapping entities, keeps the largest (outermost) span.
        """
        if not entities:
            return entities

        # Sort by span size (largest first), then by start position
        sorted_ents = sorted(
            entities, key=lambda e: (-(e['end'] - e['start']), e['start'])
        )

        kept = []
        for ent in sorted_ents:
            # Check if this entity is fully contained within any already kept entity
            is_nested = False
            for k in kept:
                if ent['start'] >= k['start'] and ent['end'] <= k['end']:
                    # ent is inside k (nested)
                    is_nested = True
                    break
            if not is_nested:
                kept.append(ent)

        # Sort back by start position for consistent ordering
        return sorted(kept, key=lambda e: e['start'])
    
    def _parse_annotations(
        self, ann_path: Path, text: str | None = None, merge_spaced_fragments: bool = True
    ) -> list[dict[str, Any]]:
        """Parse BRAT .ann file and extract entities.

        Args:
            ann_path: Path to .ann file
            text: Document text (needed for fragment merging)
            merge_spaced_fragments: If True, merge fragments separated only by whitespace
                                   (BRAT auto-splits entities at line boundaries)
        """
        entities = []

        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('T'):
                    continue

                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                # Parse entity type and span info
                # Format: "LABEL start1 end1;start2 end2;..." or "LABEL start end"
                type_info = parts[1].split()
                entity_type = type_info[0]

                # Track entity type
                self._entity_types.add(entity_type)

                # Parse ALL fragments (not just the first one)
                # Remaining parts after entity type contain the span coordinates
                span_str = ' '.join(type_info[1:])
                fragments = []

                # Parse each fragment (semicolon-separated)
                for span_part in span_str.split(';'):
                    coords = span_part.strip().split()
                    if len(coords) >= 2:
                        begin = int(coords[0])
                        end = int(coords[1])
                        fragments.append({'begin': begin, 'end': end})

                # Sort fragments by begin position
                fragments.sort(key=lambda f: f['begin'])

                # Merge fragments separated only by whitespace (like nlstruct)
                # This handles BRAT's automatic splitting at line boundaries
                if merge_spaced_fragments and text is not None and len(fragments) > 1:
                    merged_fragments = [fragments[0]]
                    for frag in fragments[1:]:
                        last_frag = merged_fragments[-1]
                        # Check if only whitespace between fragments
                        between = text[last_frag['end']:frag['begin']]
                        if between.strip() == '':
                            # Merge: extend last fragment
                            last_frag['end'] = frag['end']
                        else:
                            merged_fragments.append(frag)
                    fragments = merged_fragments

                entity_text = parts[2]

                # Use first and last fragment for overall span (backwards compat)
                # but also store all fragments for models that support discontinuous spans
                start = fragments[0]['begin']
                end = fragments[-1]['end']

                entities.append({
                    'start': start,
                    'end': end,
                    'label': entity_type,
                    'text': entity_text,
                    'fragments': fragments,  # NEW: store all fragments
                })

        return entities
    
    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        """Preprocess dataset for the task."""
        from tabib.tasks.ner_span import NERSpanTask
        
        if isinstance(task, NERSpanTask):
            task.ensure_entity_types(self.entity_types)
        
        return dataset
