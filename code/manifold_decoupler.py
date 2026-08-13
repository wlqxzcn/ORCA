from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F


VALID_ESTIMATORS = {"raw_svd", "centered_pca"}
VALID_SOLVERS = {"exact_svd", "gram_eigh"}


class SourceSubspaceProjector:

    def __init__(
        self,
        k: int = 2,
        estimator: str = "raw_svd",
        normalize_for_fit: bool = False,
        solver: str = "exact_svd",
        use_float64: bool = True,
        reorthogonalize: bool = True,
        eps: float = 1e-12,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        if estimator not in VALID_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {sorted(VALID_ESTIMATORS)}, "
                f"got {estimator!r}."
            )
        if solver not in VALID_SOLVERS:
            raise ValueError(
                f"solver must be one of {sorted(VALID_SOLVERS)}, "
                f"got {solver!r}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.k = int(k)
        self.estimator = estimator
        self.normalize_for_fit = bool(normalize_for_fit)
        self.solver = solver
        self.use_float64 = bool(use_float64)
        self.reorthogonalize = bool(reorthogonalize)
        self.eps = float(eps)

        self.basis: Optional[torch.Tensor] = None
        self.source_mean: Optional[torch.Tensor] = None
        self.singular_values: Optional[torch.Tensor] = None
        self.all_singular_values: Optional[torch.Tensor] = None
        self.total_spectral_energy: Optional[float] = None
        self.dim: Optional[int] = None
        self.k_eff: Optional[int] = None
        self.fit_metadata: Dict[str, Any] = {}


    @staticmethod
    def _ensure_2d_tensor(name: str, x: torch.Tensor) -> torch.Tensor:
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
                f"{name} must have a floating dtype, got dtype={x.dtype}."
            )
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains NaN or Inf values.")
        return x

    def _check_is_fitted(self) -> None:
        if self.basis is None or self.source_mean is None:
            raise RuntimeError("Call fit() before using SourceSubspaceProjector.")

    def _check_input(self, x: torch.Tensor) -> torch.Tensor:
        self._check_is_fitted()
        x = self._ensure_2d_tensor("x", x)
        assert self.dim is not None
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: got D={x.shape[1]}, "
                f"expected D={self.dim}."
            )
        return x

    def _basis_for(self, x: torch.Tensor) -> torch.Tensor:
        self._check_is_fitted()
        assert self.basis is not None
        return self.basis.to(device=x.device, dtype=x.dtype)

    def _mean_for(self, x: torch.Tensor) -> torch.Tensor:
        self._check_is_fitted()
        assert self.source_mean is not None
        return self.source_mean.to(device=x.device, dtype=x.dtype)


    def fit(self, source_embeddings: torch.Tensor) -> "SourceSubspaceProjector":
        """
        Fit the removable basis from source embeddings.

        No silent k truncation is performed. If k exceeds the maximum possible
        rank, a ValueError is raised so that experimental settings remain
        auditable.
        """
        source_embeddings = self._ensure_2d_tensor(
            "source_embeddings", source_embeddings
        )
        n_samples, dim = source_embeddings.shape

        if n_samples < 2:
            raise ValueError(
                "At least two source embeddings are required to estimate "
                "a subspace."
            )

        max_rank = min(n_samples, dim)
        if self.k > max_rank:
            raise ValueError(
                f"k={self.k} exceeds the maximum rank bound {max_rank} "
                f"for source shape {(n_samples, dim)}."
            )

        device = source_embeddings.device
        output_dtype = source_embeddings.dtype

        with torch.no_grad():
            source = source_embeddings.detach()

            if self.normalize_for_fit:
                source = F.normalize(
                    source, p=2, dim=1, eps=self.eps
                )

            if self.estimator == "centered_pca":
                source_mean = source.mean(dim=0, keepdim=True)
                fit_matrix = source - source_mean
            else:
                source_mean = torch.zeros(
                    (1, dim), device=source.device, dtype=source.dtype
                )
                fit_matrix = source

            spectral_dtype = (
                torch.float64 if self.use_float64 else fit_matrix.dtype
            )
            fit_matrix_spectral = fit_matrix.to(spectral_dtype)

            if self.solver == "exact_svd":
                _, singular_values, vh = torch.linalg.svd(
                    fit_matrix_spectral,
                    full_matrices=False,
                )
                basis = vh[: self.k].T.contiguous()

            elif self.solver == "gram_eigh":
                gram = fit_matrix_spectral.T @ fit_matrix_spectral
                eigenvalues, eigenvectors = torch.linalg.eigh(gram)
                order = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[order].clamp_min(0.0)
                basis = eigenvectors[:, order[: self.k]].contiguous()
                singular_values = torch.sqrt(eigenvalues)

            else:
                raise RuntimeError(f"Unsupported solver: {self.solver}")

            if self.reorthogonalize:
                basis, _ = torch.linalg.qr(basis, mode="reduced")
                basis = basis[:, : self.k]

            basis = basis.to(device=device, dtype=output_dtype)
            source_mean = source_mean.to(device=device, dtype=output_dtype)

            all_singular_values = singular_values.detach().cpu().double()
            top_singular_values = all_singular_values[: self.k]
            total_energy = float(
                torch.sum(all_singular_values ** 2).item()
            )
            top_energy = float(
                torch.sum(top_singular_values ** 2).item()
            )

            self.basis = basis
            self.source_mean = source_mean
            self.singular_values = top_singular_values
            self.all_singular_values = all_singular_values
            self.total_spectral_energy = total_energy
            self.dim = int(dim)
            self.k_eff = int(self.k)

            orthogonality_error = float(
                torch.linalg.norm(
                    basis.double().T @ basis.double()
                    - torch.eye(self.k, dtype=torch.float64, device=basis.device),
                    ord="fro",
                ).item()
            )

            self.fit_metadata = {
                "n_samples": int(n_samples),
                "dim": int(dim),
                "k": int(self.k),
                "estimator": self.estimator,
                "normalize_for_fit": self.normalize_for_fit,
                "solver": self.solver,
                "use_float64": self.use_float64,
                "reorthogonalize": self.reorthogonalize,
                "source_mean_norm": float(
                    torch.linalg.norm(source_mean.double()).item()
                ),
                "explained_energy_ratio": (
                    top_energy / total_energy
                    if total_energy > self.eps
                    else 0.0
                ),
                "orthogonality_error_fro": orthogonality_error,
                "top_singular_values": [
                    float(value)
                    for value in top_singular_values.tolist()
                ],
            }

        print(
            "[SourceSubspaceProjector] fitted: "
            f"N={n_samples}, D={dim}, k={self.k}, "
            f"estimator={self.estimator}, "
            f"normalize_for_fit={self.normalize_for_fit}, "
            f"solver={self.solver}, "
            f"energy={self.explained_energy_ratio():.6f}"
        )
        return self

    def working_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the vector representation to which the orthogonal decomposition
        is applied.

        raw_svd:
            x_work = x

        centered_pca:
            x_work = x - source_mean
        """
        x = self._check_input(x)
        if self.estimator == "centered_pca":
            return x - self._mean_for(x)
        return x

    def get_removed_component(self, x: torch.Tensor) -> torch.Tensor:
        x_work = self.working_input(x)
        basis = self._basis_for(x_work)
        return (x_work @ basis) @ basis.T

    def get_retained_component(self, x: torch.Tensor) -> torch.Tensor:
        x_work = self.working_input(x)
        removed = self.get_removed_component(x)
        return x_work - removed

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for get_retained_component()."""
        return self.get_retained_component(x)

    def decouple(
        self,
        x: torch.Tensor,
        return_raw_norms: bool = False,
        return_energy_ratios: bool = False,
    ):
        x_work = self.working_input(x)
        basis = self._basis_for(x_work)
        removed = (x_work @ basis) @ basis.T
        retained = x_work - removed

        if return_raw_norms and return_energy_ratios:
            raise ValueError(
                "Choose either return_raw_norms or return_energy_ratios, "
                "not both."
            )

        if return_raw_norms:
            retained_norm = torch.linalg.norm(retained, dim=1)
            removed_norm = torch.linalg.norm(removed, dim=1)
            return retained, removed, retained_norm, removed_norm

        if return_energy_ratios:
            total_energy = torch.sum(x_work ** 2, dim=1).clamp_min(self.eps)
            retained_energy = torch.sum(retained ** 2, dim=1)
            removed_energy = torch.sum(removed ** 2, dim=1)
            decomposition_error = (
                torch.abs(
                    total_energy - retained_energy - removed_energy
                )
                / total_energy
            )
            return (
                retained,
                removed,
                retained_energy / total_energy,
                removed_energy / total_energy,
                decomposition_error,
            )

        return retained, removed

    def removed_energy_ratio(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, removed_ratio, _ = self.decouple(
            x, return_energy_ratios=True
        )
        return removed_ratio

    def retained_energy_ratio(self, x: torch.Tensor) -> torch.Tensor:
        _, _, retained_ratio, _, _ = self.decouple(
            x, return_energy_ratios=True
        )
        return retained_ratio


    def get_projection_matrix(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        self._check_is_fitted()
        assert self.basis is not None

        basis = self.basis
        if device is not None or dtype is not None:
            basis = basis.to(
                device=device if device is not None else basis.device,
                dtype=dtype if dtype is not None else basis.dtype,
            )

        dim = basis.shape[0]
        identity = torch.eye(dim, device=basis.device, dtype=basis.dtype)
        return identity - basis @ basis.T

    def component_orthogonality_error(
        self,
        retained: torch.Tensor,
        removed: torch.Tensor,
        reduction: str = "mean",
    ) -> float:
        retained = self._ensure_2d_tensor("retained", retained)
        removed = self._ensure_2d_tensor("removed", removed)

        if retained.shape != removed.shape:
            raise ValueError(
                f"Component shape mismatch: retained={retained.shape}, "
                f"removed={removed.shape}."
            )

        numerator = torch.abs(
            torch.sum(retained * removed, dim=1)
        )
        denominator = (
            torch.linalg.norm(retained, dim=1)
            * torch.linalg.norm(removed, dim=1)
        ).clamp_min(self.eps)
        errors = numerator / denominator

        if reduction == "mean":
            return float(errors.mean().item())
        if reduction == "max":
            return float(errors.max().item())
        if reduction == "none":
            raise ValueError(
                "Use component_orthogonality_errors() for unreduced values."
            )
        raise ValueError(
            "reduction must be 'mean' or 'max'."
        )

    def component_orthogonality_errors(
        self,
        retained: torch.Tensor,
        removed: torch.Tensor,
    ) -> torch.Tensor:
        retained = self._ensure_2d_tensor("retained", retained)
        removed = self._ensure_2d_tensor("removed", removed)

        if retained.shape != removed.shape:
            raise ValueError(
                f"Component shape mismatch: retained={retained.shape}, "
                f"removed={removed.shape}."
            )

        numerator = torch.abs(
            torch.sum(retained * removed, dim=1)
        )
        denominator = (
            torch.linalg.norm(retained, dim=1)
            * torch.linalg.norm(removed, dim=1)
        ).clamp_min(self.eps)
        return numerator / denominator

    def orthogonality_error(
        self,
        x: torch.Tensor,
        reduction: str = "mean",
    ) -> float:
        retained, removed = self.decouple(x)
        return self.component_orthogonality_error(
            retained, removed, reduction=reduction
        )

    def idempotence_error(self) -> float:
        projection = self.get_projection_matrix(dtype=torch.float64)
        return float(
            torch.linalg.norm(
                projection @ projection - projection,
                ord="fro",
            ).item()
        )

    def symmetry_error(self) -> float:
        projection = self.get_projection_matrix(dtype=torch.float64)
        return float(
            torch.linalg.norm(
                projection - projection.T,
                ord="fro",
            ).item()
        )

    def explained_energy_ratio(self) -> float:
        self._check_is_fitted()
        if (
            self.singular_values is None
            or self.total_spectral_energy is None
            or self.total_spectral_energy <= self.eps
        ):
            return 0.0

        top_energy = float(
            torch.sum(self.singular_values.double() ** 2).item()
        )
        return top_energy / self.total_spectral_energy

    def per_direction_energy_ratio(self) -> torch.Tensor:
        self._check_is_fitted()
        if (
            self.singular_values is None
            or self.total_spectral_energy is None
            or self.total_spectral_energy <= self.eps
        ):
            return torch.zeros(self.k, dtype=torch.float64)

        return (
            self.singular_values.double() ** 2
            / self.total_spectral_energy
        )

    def removed_score(
        self,
        queries: torch.Tensor,
        documents: torch.Tensor,
        normalize_components: bool = False,
    ) -> torch.Tensor:
        queries = self._check_input(queries)
        documents = self._check_input(documents)
        if queries.shape[0] != documents.shape[0]:
            raise ValueError(
                "queries and documents must have the same batch size."
            )

        q_removed = self.get_removed_component(queries)
        d_removed = self.get_removed_component(documents)

        if normalize_components:
            q_removed = F.normalize(
                q_removed, p=2, dim=1, eps=self.eps
            )
            d_removed = F.normalize(
                d_removed, p=2, dim=1, eps=self.eps
            )

        return torch.sum(q_removed * d_removed, dim=1)

    def retained_score(
        self,
        queries: torch.Tensor,
        documents: torch.Tensor,
        normalize_components: bool = False,
    ) -> torch.Tensor:
        queries = self._check_input(queries)
        documents = self._check_input(documents)
        if queries.shape[0] != documents.shape[0]:
            raise ValueError(
                "queries and documents must have the same batch size."
            )

        q_retained = self.get_retained_component(queries)
        d_retained = self.get_retained_component(documents)

        if normalize_components:
            q_retained = F.normalize(
                q_retained, p=2, dim=1, eps=self.eps
            )
            d_retained = F.normalize(
                d_retained, p=2, dim=1, eps=self.eps
            )

        return torch.sum(q_retained * d_retained, dim=1)


    def state_dict(self) -> Dict[str, Any]:
        self._check_is_fitted()
        assert self.basis is not None
        assert self.source_mean is not None

        return {
            "config": {
                "k": self.k,
                "estimator": self.estimator,
                "normalize_for_fit": self.normalize_for_fit,
                "solver": self.solver,
                "use_float64": self.use_float64,
                "reorthogonalize": self.reorthogonalize,
                "eps": self.eps,
            },
            "basis": self.basis.detach().cpu(),
            "source_mean": self.source_mean.detach().cpu(),
            "singular_values": (
                None
                if self.singular_values is None
                else self.singular_values.detach().cpu()
            ),
            "all_singular_values": (
                None
                if self.all_singular_values is None
                else self.all_singular_values.detach().cpu()
            ),
            "total_spectral_energy": self.total_spectral_energy,
            "dim": self.dim,
            "k_eff": self.k_eff,
            "fit_metadata": self.fit_metadata,
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SourceSubspaceProjector":
        config = state.get("config", {})
        self.k = int(config.get("k", self.k))
        self.estimator = str(
            config.get("estimator", self.estimator)
        )
        self.normalize_for_fit = bool(
            config.get("normalize_for_fit", self.normalize_for_fit)
        )
        self.solver = str(config.get("solver", self.solver))
        self.use_float64 = bool(
            config.get("use_float64", self.use_float64)
        )
        self.reorthogonalize = bool(
            config.get("reorthogonalize", self.reorthogonalize)
        )
        self.eps = float(config.get("eps", self.eps))

        basis = state["basis"]
        source_mean = state["source_mean"]

        if device is not None or dtype is not None:
            basis = basis.to(
                device=device if device is not None else basis.device,
                dtype=dtype if dtype is not None else basis.dtype,
            )
            source_mean = source_mean.to(
                device=device if device is not None else source_mean.device,
                dtype=dtype if dtype is not None else source_mean.dtype,
            )

        self.basis = basis
        self.source_mean = source_mean
        self.singular_values = state.get("singular_values")
        self.all_singular_values = state.get("all_singular_values")
        self.total_spectral_energy = state.get(
            "total_spectral_energy"
        )
        self.dim = int(state["dim"])
        self.k_eff = int(state["k_eff"])
        self.fit_metadata = dict(state.get("fit_metadata", {}))
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SourceSubspaceProjector":
        state = torch.load(
            path,
            map_location=device if device is not None else "cpu",
        )
        config = state.get("config", {})
        projector = cls(
            k=int(config.get("k", 2)),
            estimator=str(config.get("estimator", "raw_svd")),
            normalize_for_fit=bool(
                config.get("normalize_for_fit", False)
            ),
            solver=str(config.get("solver", "exact_svd")),
            use_float64=bool(config.get("use_float64", True)),
            reorthogonalize=bool(
                config.get("reorthogonalize", True)
            ),
            eps=float(config.get("eps", 1e-12)),
        )
        return projector.load_state_dict(
            state, device=device, dtype=dtype
        )


    @property
    def emotion_basis(self) -> Optional[torch.Tensor]:
        warnings.warn(
            "emotion_basis is deprecated; use basis instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.basis

    @property
    def U_k(self) -> Optional[torch.Tensor]:
        warnings.warn(
            "U_k is deprecated and was mathematically misnamed; "
            "use basis instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.basis

    def get_emotion_component(self, x: torch.Tensor) -> torch.Tensor:
        warnings.warn(
            "get_emotion_component() is deprecated; "
            "use get_removed_component().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_removed_component(x)

    def remove_subspace(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_retained_component(x)


class ManifoldDecoupler(SourceSubspaceProjector):

    def __init__(
        self,
        k: int = 2,
        normalize_for_svd: bool = False,
        center_for_svd: bool = False,
        use_float64_svd: bool = True,
        reorthogonalize: bool = True,
        eps: float = 1e-12,
        solver: str = "exact_svd",
    ) -> None:
        estimator = (
            "centered_pca" if center_for_svd else "raw_svd"
        )
        super().__init__(
            k=k,
            estimator=estimator,
            normalize_for_fit=normalize_for_svd,
            solver=solver,
            use_float64=use_float64_svd,
            reorthogonalize=reorthogonalize,
            eps=eps,
        )

        self.normalize_for_svd = normalize_for_svd
        self.center_for_svd = center_for_svd
        self.use_float64_svd = use_float64_svd
