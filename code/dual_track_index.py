from __future__ import annotations
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from manifold_decoupler import (
    ManifoldDecoupler,
    SourceSubspaceProjector,
)


VALID_SCORE_METRICS = {"cosine", "inner_product"}


class SymmetricDualTrackIndex:

    def __init__(
        self,
        projector: SourceSubspaceProjector,
        score_metric: str = "cosine",
    ) -> None:
        if not isinstance(
            projector, SourceSubspaceProjector
        ):
            raise TypeError(
                "projector must be a SourceSubspaceProjector "
                "or ManifoldDecoupler."
            )
        if score_metric not in VALID_SCORE_METRICS:
            raise ValueError(
                f"score_metric must be one of "
                f"{sorted(VALID_SCORE_METRICS)}, "
                f"got {score_metric!r}."
            )

        self.projector = projector
        # Compatibility alias used by old code.
        self.decoupler = projector

        self.score_metric = score_metric
        self.normalize = score_metric == "cosine"

        self.task_index: Optional[faiss.Index] = None
        self.task_docs: List[str] = []

        self.removed_index: Optional[faiss.Index] = None
        self.removed_docs: List[str] = []

        self.dim: Optional[int] = None
        self.index_size: int = 0


    @staticmethod
    def _ensure_2d_tensor(
        name: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(
                f"{name} must have shape (D,) or (N, D), "
                f"got shape={tuple(x.shape)}."
            )
        if not torch.is_floating_point(x):
            raise TypeError(
                f"{name} must have a floating dtype, "
                f"got dtype={x.dtype}."
            )
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains NaN or Inf values.")
        return x

    @staticmethod
    def _to_faiss_np(x: torch.Tensor) -> np.ndarray:
        array = (
            x.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("float32", copy=False)
        )
        return np.ascontiguousarray(array)

    def _prepare_for_search(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.score_metric == "cosine":
            return F.normalize(
                x, p=2, dim=1, eps=self.projector.eps
            )
        return x

    @staticmethod
    def _validate_texts(
        name: str,
        texts: Sequence[str],
        expected_size: int,
    ) -> List[str]:
        texts = [str(text) for text in texts]
        if len(texts) != expected_size:
            raise ValueError(
                f"len({name})={len(texts)} does not match "
                f"embedding count={expected_size}."
            )
        return texts

    @staticmethod
    def _new_flat_index(
        dim: int,
    ) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(dim)


    def build_index(
        self,
        raw_task_embeddings: torch.Tensor,
        task_texts: Sequence[str],
        raw_removed_embeddings: Optional[torch.Tensor] = None,
        removed_texts: Optional[Sequence[str]] = None,
        score_metric: Optional[str] = None,
    ) -> "SymmetricDualTrackIndex":
        if score_metric is not None:
            if score_metric not in VALID_SCORE_METRICS:
                raise ValueError(
                    f"score_metric must be one of "
                    f"{sorted(VALID_SCORE_METRICS)}."
                )
            self.score_metric = score_metric
            self.normalize = score_metric == "cosine"

        raw_task_embeddings = self._ensure_2d_tensor(
            "raw_task_embeddings", raw_task_embeddings
        )
        if raw_task_embeddings.shape[0] == 0:
            raise ValueError(
                "raw_task_embeddings must contain at least one row."
            )

        task_texts_list = self._validate_texts(
            "task_texts",
            task_texts,
            raw_task_embeddings.shape[0],
        )

        self.dim = int(raw_task_embeddings.shape[1])

        retained_embeddings = (
            self.projector.get_retained_component(
                raw_task_embeddings
            )
        )
        retained_embeddings = self._prepare_for_search(
            retained_embeddings
        )
        retained_np = self._to_faiss_np(retained_embeddings)

        self.task_index = self._new_flat_index(
            retained_np.shape[1]
        )
        self.task_index.add(retained_np)
        self.task_docs = task_texts_list
        self.index_size = len(self.task_docs)

        self.removed_index = None
        self.removed_docs = []

        if (
            raw_removed_embeddings is None
            and removed_texts is None
        ):
            pass
        elif (
            raw_removed_embeddings is None
            or removed_texts is None
        ):
            raise ValueError(
                "raw_removed_embeddings and removed_texts "
                "must be provided together."
            )
        else:
            raw_removed_embeddings = self._ensure_2d_tensor(
                "raw_removed_embeddings",
                raw_removed_embeddings,
            )
            if raw_removed_embeddings.shape[1] != self.dim:
                raise ValueError(
                    "Removed-space embedding dimension mismatch: "
                    f"got D={raw_removed_embeddings.shape[1]}, "
                    f"expected D={self.dim}."
                )

            removed_texts_list = self._validate_texts(
                "removed_texts",
                removed_texts,
                raw_removed_embeddings.shape[0],
            )

            removed_embeddings = (
                self.projector.get_removed_component(
                    raw_removed_embeddings
                )
            )
            removed_embeddings = self._prepare_for_search(
                removed_embeddings
            )
            removed_np = self._to_faiss_np(
                removed_embeddings
            )

            self.removed_index = self._new_flat_index(
                removed_np.shape[1]
            )
            self.removed_index.add(removed_np)
            self.removed_docs = removed_texts_list

        print(
            "[SymmetricDualTrackIndex] built: "
            f"task={len(self.task_docs)}, "
            f"removed={len(self.removed_docs)}, "
            f"metric={self.score_metric}, "
            f"dim={self.dim}"
        )
        return self

    def build_offline_index(
        self,
        raw_med_embs: torch.Tensor,
        med_texts: Sequence[str],
        raw_mem_embs: Optional[torch.Tensor] = None,
        mem_texts: Optional[Sequence[str]] = None,
        normalize: bool = True,
    ) -> "SymmetricDualTrackIndex":
        warnings.warn(
            "build_offline_index() uses deprecated med/mem terminology. "
            "Use build_index() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        metric = "cosine" if normalize else "inner_product"
        return self.build_index(
            raw_task_embeddings=raw_med_embs,
            task_texts=med_texts,
            raw_removed_embeddings=raw_mem_embs,
            removed_texts=mem_texts,
            score_metric=metric,
        )


    @staticmethod
    def _search_index(
        index: Optional[faiss.Index],
        query_embeddings: torch.Tensor,
        top_k: int,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        if index is None or top_k <= 0:
            batch_size = int(query_embeddings.shape[0])
            return (
                [[] for _ in range(batch_size)],
                [[] for _ in range(batch_size)],
            )

        k = min(int(top_k), int(index.ntotal))
        if k <= 0:
            batch_size = int(query_embeddings.shape[0])
            return (
                [[] for _ in range(batch_size)],
                [[] for _ in range(batch_size)],
            )

        query_np = (
            query_embeddings.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("float32", copy=False)
        )
        query_np = np.ascontiguousarray(query_np)

        scores, ids = index.search(query_np, k)

        id_rows = [
            [int(value) for value in row.tolist()]
            for row in ids
        ]
        score_rows = [
            [float(value) for value in row.tolist()]
            for row in scores
        ]
        return id_rows, score_rows

    @staticmethod
    def _texts_from_ids(
        ids: Sequence[Sequence[int]],
        docs: Sequence[str],
    ) -> List[List[str]]:
        return [
            [docs[index] for index in row]
            for row in ids
        ]


    def retrieve_batch(
        self,
        query_embeddings: torch.Tensor,
        task_top_k: int = 5,
        removed_top_k: int = 0,
        return_orthogonality: bool = False,
        return_removed_energy_ratio: bool = False,
    ) -> Dict[str, Any]:
        if self.task_index is None:
            raise RuntimeError(
                "Call build_index() before retrieve_batch()."
            )
        if self.dim is None:
            raise RuntimeError("Index dimension is not initialized.")

        query_embeddings = self._ensure_2d_tensor(
            "query_embeddings", query_embeddings
        )
        if query_embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension mismatch: "
                f"got D={query_embeddings.shape[1]}, "
                f"expected D={self.dim}."
            )
        if task_top_k < 0 or removed_top_k < 0:
            raise ValueError(
                "task_top_k and removed_top_k must be non-negative."
            )

        start_time = time.perf_counter()

        (
            retained_queries,
            removed_queries,
            _,
            removed_ratios,
            decomposition_errors,
        ) = self.projector.decouple(
            query_embeddings,
            return_energy_ratios=True,
        )

        orthogonality_errors: Optional[torch.Tensor] = None
        if return_orthogonality:
            orthogonality_errors = (
                self.projector.component_orthogonality_errors(
                    retained_queries,
                    removed_queries,
                )
            )

        retained_queries_search = self._prepare_for_search(
            retained_queries
        )
        removed_queries_search = self._prepare_for_search(
            removed_queries
        )

        task_ids, task_scores = self._search_index(
            self.task_index,
            retained_queries_search,
            task_top_k,
        )
        task_texts = self._texts_from_ids(
            task_ids, self.task_docs
        )

        removed_ids, removed_scores = self._search_index(
            self.removed_index,
            removed_queries_search,
            removed_top_k,
        )
        removed_texts = self._texts_from_ids(
            removed_ids, self.removed_docs
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0
        batch_size = int(query_embeddings.shape[0])

        result: Dict[str, Any] = {
            "task_ids": task_ids,
            "task_scores": task_scores,
            "task_texts": task_texts,
            "removed_ids": removed_ids,
            "removed_scores": removed_scores,
            "removed_texts": removed_texts,
            "batch_size": batch_size,
            "latency_ms_total": elapsed_ms,
            "latency_ms_per_query": elapsed_ms / max(
                batch_size, 1
            ),
            "score_metric": self.score_metric,
        }

        if return_orthogonality:
            assert orthogonality_errors is not None
            result["orthogonality_errors"] = [
                float(value)
                for value in orthogonality_errors.detach()
                .cpu()
                .tolist()
            ]

        if return_removed_energy_ratio:
            result["removed_energy_ratios"] = [
                float(value)
                for value in removed_ratios.detach()
                .cpu()
                .tolist()
            ]
            result["decomposition_errors"] = [
                float(value)
                for value in decomposition_errors.detach()
                .cpu()
                .tolist()
            ]

        return result


    def retrieve(
        self,
        query_emb: torch.Tensor,
        med_top_k: int = 1,
        mem_top_k: int = 1,
        return_ortho: bool = True,
        return_emotion_intensity: bool = False,
        return_removed_energy_ratio: Optional[bool] = None,
    ) -> Dict[str, Any]:
        query_emb = self._ensure_2d_tensor(
            "query_emb", query_emb
        )
        if query_emb.shape[0] != 1:
            raise ValueError(
                "retrieve() accepts one query only. "
                "Use retrieve_batch() for multiple queries."
            )

        if return_removed_energy_ratio is None:
            return_removed_energy_ratio = (
                return_emotion_intensity
            )

        batch_result = self.retrieve_batch(
            query_embeddings=query_emb,
            task_top_k=med_top_k,
            removed_top_k=mem_top_k,
            return_orthogonality=return_ortho,
            return_removed_energy_ratio=(
                return_removed_energy_ratio
            ),
        )

        result: Dict[str, Any] = {
            "task_ids": batch_result["task_ids"][0],
            "task_scores": batch_result["task_scores"][0],
            "task_texts": batch_result["task_texts"][0],
            "removed_ids": batch_result["removed_ids"][0],
            "removed_scores": batch_result[
                "removed_scores"
            ][0],
            "removed_texts": batch_result[
                "removed_texts"
            ][0],
            "latency_ms": batch_result[
                "latency_ms_total"
            ],
            "score_metric": batch_result[
                "score_metric"
            ],
        }

        result["med_ids"] = result["task_ids"]
        result["med_scores"] = result["task_scores"]
        result["med_texts"] = result["task_texts"]

        result["mem_ids"] = result["removed_ids"]
        result["mem_scores"] = result["removed_scores"]
        result["mem_texts"] = result["removed_texts"]

        if return_ortho:
            result["ortho_error"] = batch_result[
                "orthogonality_errors"
            ][0]

        if return_removed_energy_ratio:
            ratio = batch_result[
                "removed_energy_ratios"
            ][0]
            result["removed_energy_ratio"] = ratio

            if return_emotion_intensity:
                warnings.warn(
                    "emotion_intensity is deprecated and now aliases "
                    "removed_energy_ratio. It must not be interpreted "
                    "as a validated emotion score.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                result["emotion_intensity"] = ratio

        return result

    def retrieve_task_only(
        self,
        query_emb: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return self.retrieve(
            query_emb=query_emb,
            med_top_k=top_k,
            mem_top_k=0,
            return_ortho=False,
            return_emotion_intensity=False,
            return_removed_energy_ratio=False,
        )

    def retrieve_removed_only(
        self,
        query_emb: torch.Tensor,
        top_k: int = 5,
        return_removed_energy_ratio: bool = True,
    ) -> Dict[str, Any]:
        return self.retrieve(
            query_emb=query_emb,
            med_top_k=0,
            mem_top_k=top_k,
            return_ortho=False,
            return_emotion_intensity=False,
            return_removed_energy_ratio=(
                return_removed_energy_ratio
            ),
        )

    def retrieve_emotion_only(
        self,
        query_emb: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        warnings.warn(
            "retrieve_emotion_only() is deprecated; use "
            "retrieve_removed_only(). The removed subspace is not "
            "a validated emotion representation.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.retrieve(
            query_emb=query_emb,
            med_top_k=0,
            mem_top_k=top_k,
            return_ortho=False,
            return_emotion_intensity=True,
        )


    @property
    def med_index(self) -> Optional[faiss.Index]:
        warnings.warn(
            "med_index is deprecated; use task_index.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.task_index

    @property
    def med_docs(self) -> List[str]:
        warnings.warn(
            "med_docs is deprecated; use task_docs.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.task_docs

    @property
    def mem_embeddings(self):
        warnings.warn(
            "mem_embeddings is no longer stored directly. "
            "Use removed_index.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    @property
    def mem_docs(self) -> List[str]:
        warnings.warn(
            "mem_docs is deprecated; use removed_docs.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.removed_docs
