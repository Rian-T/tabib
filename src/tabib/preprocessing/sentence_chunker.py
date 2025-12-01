"""Sentence-based chunker for long documents."""

import re
from typing import Any

from datasets import Dataset

from tabib.preprocessing.base import Preprocessor


class SentenceChunker(Preprocessor):
    """Chunk documents into smaller pieces based on sentence boundaries.
    
    Fills chunks with complete sentences until adding another would
    exceed the context window. This avoids splitting entities.
    """
    
    def __init__(self, tokenizer=None):
        """Initialize chunker.
        
        Args:
            tokenizer: Optional tokenizer for accurate token counting
        """
        self.tokenizer = tokenizer
    
    def preprocess(self, dataset: Dataset, max_length: int) -> Dataset:
        """Chunk documents that exceed max_length.
        
        Args:
            dataset: Dataset with 'text', 'doc_id', and 'entities' fields
            max_length: Maximum tokens per chunk
            
        Returns:
            Dataset with chunked documents and metadata
        """
        chunked_examples = []
        
        # Reset format to access raw data
        if hasattr(dataset, 'set_format'):
            dataset = dataset.with_format(None)
        
        for example in dataset:
            doc_id = example['doc_id']
            text = example['text']
            entities = example['entities']
            
            # Estimate if chunking needed (rough: 4 chars per token)
            if len(text) < max_length * 4:
                # No chunking needed
                chunked_examples.append({
                    'doc_id': doc_id,
                    'chunk_id': 0,
                    'text': text,
                    'entities': entities,
                    'chunk_offset': 0
                })
            else:
                # Chunk the document
                chunks = self._chunk_document(text, entities, max_length)
                for chunk_id, chunk in enumerate(chunks):
                    chunked_examples.append({
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'text': chunk['text'],
                        'entities': chunk['entities'],
                        'chunk_offset': chunk['offset']
                    })
        
        return Dataset.from_list(chunked_examples)
    
    def _chunk_document(
        self, text: str, entities: list[dict], max_length: int
    ) -> list[dict]:
        """Chunk a document into smaller pieces."""
        sentences = self._split_sentences(text)
        chunks = []
        
        current_text = []
        current_start = 0
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check if adding this sentence would exceed limit
            if current_text and current_length + sentence_tokens > max_length:
                # Save current chunk
                chunk_text = ''.join(current_text)
                chunk_end = current_start + len(chunk_text)
                chunk_entities = self._filter_entities(
                    entities, current_start, chunk_end
                )
                chunks.append({
                    'text': chunk_text,
                    'entities': chunk_entities,
                    'offset': current_start
                })
                
                # Start new chunk
                current_text = [sentence]
                current_start = chunk_end
                current_length = sentence_tokens
            else:
                current_text.append(sentence)
                current_length += sentence_tokens
        
        # Add final chunk
        if current_text:
            chunk_text = ''.join(current_text)
            chunk_end = current_start + len(chunk_text)
            chunk_entities = self._filter_entities(
                entities, current_start, chunk_end
            )
            chunks.append({
                'text': chunk_text,
                'entities': chunk_entities,
                'offset': current_start
            })
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, preserving delimiters."""
        # Simple sentence splitting on common punctuation
        pattern = r'([.!?]+\s+)'
        parts = re.split(pattern, text)
        
        sentences = []
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]
            if sentence:
                sentences.append(sentence)
        
        return sentences
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Rough estimate: 4 chars per token
        return len(text) // 4
    
    def _filter_entities(
        self, entities: list[dict], chunk_start: int, chunk_end: int
    ) -> list[dict]:
        """Filter entities that fall within chunk boundaries."""
        chunk_entities = []
        
        for entity in entities:
            ent_start = entity['start']
            ent_end = entity['end']
            
            # Include entity if it's completely within chunk
            if ent_start >= chunk_start and ent_end <= chunk_end:
                # Adjust offsets to be chunk-relative
                chunk_entities.append({
                    'start': ent_start - chunk_start,
                    'end': ent_end - chunk_start,
                    'label': entity['label'],
                    'text': entity['text']
                })
        
        return chunk_entities

