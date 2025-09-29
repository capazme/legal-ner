"""
Semantic Correlator Service
===========================

Utilizza distil-bert come sentence transformer per correlare semanticamente
riferimenti normativi distanti nel testo.

Esempio:
Paragrafo 1: "Il decreto legislativo n. 231 del 2001..."
Paragrafo 3: "L'articolo 25 disciplina..."
→ Correlazione: art. 25 appartiene al D.Lgs. 231/2001

Questo sistema risolve la frammentazione dei riferimenti legali nei documenti.
"""

from typing import List, Dict, Any, Tuple, Optional
import structlog
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from dataclasses import dataclass

log = structlog.get_logger()

@dataclass
class SemanticChunk:
    """Chunk di testo con embeddings per correlazione semantica."""
    text: str
    start_char: int
    end_char: int
    embedding: np.ndarray
    entities: List[Dict[str, Any]]
    chunk_id: str

@dataclass
class CorrelationResult:
    """Risultato di una correlazione semantica."""
    primary_entity: Dict[str, Any]
    correlated_entity: Dict[str, Any]
    correlation_score: float
    distance_chars: int
    explanation: str

class SemanticCorrelator:
    """
    Correlatore semantico per riferimenti normativi usando sentence transformers.

    Funzionalità:
    1. Chunking intelligente del testo
    2. Embedding generation con distil-bert
    3. Correlazione semantica tra chunks
    4. Associazione entità distanti ma correlate
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        log.info("Inizializzazione SemanticCorrelator", model=model_name)
        self.sentence_transformer = SentenceTransformer(model_name)
        self.chunk_size = 200  # Caratteri per chunk
        self.overlap_size = 50  # Overlap tra chunks
        self.correlation_threshold = 0.75  # Soglia similarità semantica

    async def correlate_legal_references(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[CorrelationResult]]:
        """
        Correla riferimenti normativi distanti nel testo usando embeddings semantici.

        Args:
            text: Testo completo del documento
            entities: Entità estratte dai 3 stagi

        Returns:
            Tuple[enhanced_entities, correlation_results]
        """
        log.info("Starting semantic correlation", text_length=len(text), entities_count=len(entities))

        # 1. Chunking intelligente del testo
        chunks = self._create_semantic_chunks(text, entities)
        log.debug("Text chunked", chunks_count=len(chunks))

        # 2. Generate embeddings per ogni chunk
        await self._generate_chunk_embeddings(chunks)
        log.debug("Embeddings generated")

        # 3. Trova correlazioni semantiche
        correlations = await self._find_semantic_correlations(chunks)
        log.debug("Correlations found", correlations_count=len(correlations))

        # 4. Enhance entities con correlazioni
        enhanced_entities = await self._enhance_entities_with_correlations(entities, correlations)

        log.info("Semantic correlation complete",
                enhanced_entities=len(enhanced_entities),
                correlations=len(correlations))

        return enhanced_entities, correlations

    def _create_semantic_chunks(self, text: str, entities: List[Dict[str, Any]]) -> List[SemanticChunk]:
        """
        Crea chunks semantici del testo, considerando le entità presenti.
        """
        chunks = []

        # Sliding window chunking con overlap
        for i in range(0, len(text), self.chunk_size - self.overlap_size):
            chunk_start = i
            chunk_end = min(i + self.chunk_size, len(text))
            chunk_text = text[chunk_start:chunk_end]

            # Trova entità in questo chunk
            chunk_entities = []
            for entity in entities:
                if (entity["start_char"] >= chunk_start and
                    entity["end_char"] <= chunk_end):
                    chunk_entities.append(entity)

            chunk = SemanticChunk(
                text=chunk_text,
                start_char=chunk_start,
                end_char=chunk_end,
                embedding=None,  # Generato dopo
                entities=chunk_entities,
                chunk_id=f"chunk_{i//100}"
            )
            chunks.append(chunk)

        log.debug("Created semantic chunks",
                 chunks_count=len(chunks),
                 avg_entities_per_chunk=np.mean([len(c.entities) for c in chunks]))

        return chunks

    async def _generate_chunk_embeddings(self, chunks: List[SemanticChunk]):
        """
        Genera embeddings per ogni chunk usando sentence transformer.
        """
        chunk_texts = [chunk.text for chunk in chunks]

        # Batch encoding per efficiency
        embeddings = self.sentence_transformer.encode(
            chunk_texts,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Assegna embeddings ai chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

    async def _find_semantic_correlations(self, chunks: List[SemanticChunk]) -> List[CorrelationResult]:
        """
        Trova correlazioni semantiche tra chunks usando cosine similarity.
        """
        correlations = []

        # Identifica chunks con entità legali rilevanti
        legal_chunks = []
        article_chunks = []

        for chunk in chunks:
            has_legal_ref = any(entity["label"] in ["LEGGE", "N_SENTENZA"] for entity in chunk.entities)
            has_article_ref = any("art" in entity["text"].lower() or
                                 self._is_article_reference(entity["text"])
                                 for entity in chunk.entities)

            if has_legal_ref:
                legal_chunks.append(chunk)
            if has_article_ref:
                article_chunks.append(chunk)

        log.debug("Identified correlation candidates",
                 legal_chunks=len(legal_chunks),
                 article_chunks=len(article_chunks))

        # Trova correlazioni tra legal refs e article refs
        for legal_chunk in legal_chunks:
            for article_chunk in article_chunks:
                if legal_chunk.chunk_id == article_chunk.chunk_id:
                    continue  # Skip stesso chunk

                # Calcola similarità semantica
                similarity = cosine_similarity(
                    legal_chunk.embedding.reshape(1, -1),
                    article_chunk.embedding.reshape(1, -1)
                )[0][0]

                if similarity >= self.correlation_threshold:
                    # Crea correlazione
                    correlation = await self._create_correlation(
                        legal_chunk, article_chunk, similarity
                    )
                    if correlation:
                        correlations.append(correlation)

        return correlations

    def _is_article_reference(self, text: str) -> bool:
        """Verifica se il testo contiene un riferimento ad articolo."""
        article_patterns = [
            r"\bart\.?\s*\d+",
            r"\barticolo\s+\d+",
            r"\bcomma\s+\d+",
            r"\blettera\s+[a-z]"
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in article_patterns)

    async def _create_correlation(
        self,
        legal_chunk: SemanticChunk,
        article_chunk: SemanticChunk,
        similarity: float
    ) -> Optional[CorrelationResult]:
        """
        Crea una correlazione specifica tra chunk con riferimento legale e articolo.
        """
        # Trova la migliore entità legale nel legal_chunk
        legal_entity = None
        for entity in legal_chunk.entities:
            if entity["label"] in ["LEGGE", "N_SENTENZA"]:
                legal_entity = entity
                break

        # Trova la migliore entità articolo nel article_chunk
        article_entity = None
        for entity in article_chunk.entities:
            if (self._is_article_reference(entity["text"]) or
                "art" in entity["text"].lower()):
                article_entity = entity
                break

        if not legal_entity or not article_entity:
            return None

        # Calcola distanza in caratteri
        distance_chars = abs(legal_chunk.start_char - article_chunk.start_char)

        # Genera spiegazione
        explanation = f"Correlazione semantica tra '{legal_entity['text']}' e '{article_entity['text']}' (similarità: {similarity:.2f})"

        return CorrelationResult(
            primary_entity=legal_entity,
            correlated_entity=article_entity,
            correlation_score=similarity,
            distance_chars=distance_chars,
            explanation=explanation
        )

    async def _enhance_entities_with_correlations(
        self,
        entities: List[Dict[str, Any]],
        correlations: List[CorrelationResult]
    ) -> List[Dict[str, Any]]:
        """
        Arricchisce le entità con informazioni di correlazione.
        """
        enhanced_entities = []

        for entity in entities:
            enhanced_entity = entity.copy()

            # Trova correlazioni per questa entità
            entity_correlations = []
            for correlation in correlations:
                if (correlation.primary_entity["text"] == entity["text"] or
                    correlation.correlated_entity["text"] == entity["text"]):
                    entity_correlations.append({
                        "correlated_with": (correlation.correlated_entity["text"]
                                          if correlation.primary_entity["text"] == entity["text"]
                                          else correlation.primary_entity["text"]),
                        "correlation_score": correlation.correlation_score,
                        "distance_chars": correlation.distance_chars,
                        "explanation": correlation.explanation
                    })

            # Aggiungi correlazioni all'entità se presenti
            if entity_correlations:
                enhanced_entity["semantic_correlations"] = entity_correlations
                enhanced_entity["has_correlations"] = True

                # Boost confidence per entità correlate
                original_confidence = enhanced_entity["confidence"]
                correlation_boost = min(0.1, len(entity_correlations) * 0.03)
                enhanced_entity["confidence"] = min(0.99, original_confidence + correlation_boost)

                log.debug("Entity enhanced with correlations",
                         entity=entity["text"],
                         correlations_count=len(entity_correlations),
                         confidence_boost=correlation_boost)

            enhanced_entities.append(enhanced_entity)

        return enhanced_entities

    async def correlate_article_to_law(
        self,
        article_entity: Dict[str, Any],
        text: str,
        law_entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Trova la legge più probabilmente associata a un articolo usando similarità semantica.

        Caso d'uso specifico: "art. 25" → "D.Lgs. 231/2001"
        """
        if not law_entities:
            return None

        # Context window attorno all'articolo
        context_start = max(0, article_entity["start_char"] - 300)
        context_end = min(len(text), article_entity["end_char"] + 300)
        article_context = text[context_start:context_end]

        # Genera embedding per il contesto dell'articolo
        article_embedding = self.sentence_transformer.encode([article_context])[0]

        best_law = None
        best_similarity = 0.0

        for law_entity in law_entities:
            # Context window attorno alla legge
            law_context_start = max(0, law_entity["start_char"] - 100)
            law_context_end = min(len(text), law_entity["end_char"] + 100)
            law_context = text[law_context_start:law_context_end]

            # Genera embedding per il contesto della legge
            law_embedding = self.sentence_transformer.encode([law_context])[0]

            # Calcola similarità
            similarity = cosine_similarity(
                article_embedding.reshape(1, -1),
                law_embedding.reshape(1, -1)
            )[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_law = law_entity

        # Ritorna la legge correlata se sopra soglia
        if best_similarity >= self.correlation_threshold:
            return {
                "correlated_law": best_law,
                "correlation_score": best_similarity,
                "explanation": f"Articolo correlato semanticamente a {best_law['text']}"
            }

        return None

    def get_correlation_stats(self, correlations: List[CorrelationResult]) -> Dict[str, Any]:
        """Statistiche sulle correlazioni trovate."""
        if not correlations:
            return {"total_correlations": 0}

        scores = [c.correlation_score for c in correlations]
        distances = [c.distance_chars for c in correlations]

        return {
            "total_correlations": len(correlations),
            "avg_correlation_score": np.mean(scores),
            "min_correlation_score": np.min(scores),
            "max_correlation_score": np.max(scores),
            "avg_distance_chars": np.mean(distances),
            "min_distance_chars": np.min(distances),
            "max_distance_chars": np.max(distances),
            "correlation_types": self._analyze_correlation_types(correlations)
        }

    def _analyze_correlation_types(self, correlations: List[CorrelationResult]) -> Dict[str, int]:
        """Analizza i tipi di correlazioni trovate."""
        types = {}
        for correlation in correlations:
            primary_type = correlation.primary_entity["label"]
            correlated_type = correlation.correlated_entity["label"]
            correlation_type = f"{primary_type}->{correlated_type}"
            types[correlation_type] = types.get(correlation_type, 0) + 1
        return types