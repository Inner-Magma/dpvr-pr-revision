"""Unit tests for CKA and KL-divergence analysis utilities."""
from __future__ import annotations

import torch
import pytest

from dpvr.analysis.cka import linear_cka
from dpvr.analysis.kl_divergence import kl_divergence_logits


# ---------------------------------------------------------------------------
# CKA tests
# ---------------------------------------------------------------------------

def test_linear_cka_self_similarity():
    """CKA(X, X) must equal 1.0 for any non-degenerate matrix."""
    torch.manual_seed(0)
    X = torch.randn(32, 64)
    score = linear_cka(X, X)
    assert abs(score - 1.0) < 1e-5, f"Expected CKA(X,X)=1.0, got {score}"


def test_linear_cka_range():
    """CKA must lie in [0, 1]."""
    torch.manual_seed(1)
    X = torch.randn(32, 64)
    Y = torch.randn(32, 128)
    score = linear_cka(X, Y)
    assert 0.0 <= score <= 1.0 + 1e-6, f"CKA out of range: {score}"


def test_linear_cka_symmetric():
    """CKA(X, Y) == CKA(Y, X)."""
    torch.manual_seed(2)
    X = torch.randn(16, 32)
    Y = torch.randn(16, 64)
    assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-5


def test_linear_cka_uncorrelated_less_than_self():
    """CKA of two independent random matrices should be well below 1.0."""
    torch.manual_seed(6)
    # Two independent random matrices with very different structure
    X = torch.randn(32, 64)
    # Y is in a completely different random direction
    Y = torch.randn(32, 64) * 10 + torch.randn(32, 64)
    score_xy = linear_cka(X, Y)
    score_xx = linear_cka(X, X)  # = 1.0
    assert score_xy < score_xx, f"Cross-CKA {score_xy} should be < self-CKA {score_xx}"


# ---------------------------------------------------------------------------
# KL divergence tests
# ---------------------------------------------------------------------------

def test_kl_self_is_zero():
    """KL(P || P) must be 0.0 for any distribution."""
    torch.manual_seed(3)
    logits = torch.randn(2, 10, 1000)
    kl = kl_divergence_logits(logits, logits).item()
    assert abs(kl) < 1e-4, f"Expected KL(P||P)=0, got {kl}"


def test_kl_asymmetric():
    """KL divergence is asymmetric: KL(P||Q) != KL(Q||P) in general."""
    torch.manual_seed(4)
    p_logits = torch.randn(1, 5, 100)
    q_logits = torch.randn(1, 5, 100)
    kl_pq = kl_divergence_logits(p_logits, q_logits).item()
    kl_qp = kl_divergence_logits(q_logits, p_logits).item()
    # They should differ for random logits.
    assert abs(kl_pq - kl_qp) > 1e-4, "KL should be asymmetric for random logits"


def test_kl_nonnegative():
    """KL divergence is always >= 0."""
    torch.manual_seed(5)
    logits_p = torch.randn(2, 8, 512)
    logits_q = torch.randn(2, 8, 512)
    kl = kl_divergence_logits(logits_p, logits_q).item()
    assert kl >= -1e-5, f"KL divergence must be non-negative, got {kl}"


def test_kl_reduction_none_shape():
    """With reduction='none', output shape is (B, T)."""
    B, T, V = 3, 7, 256
    p = torch.randn(B, T, V)
    q = torch.randn(B, T, V)
    out = kl_divergence_logits(p, q, reduction="none")
    assert out.shape == (B, T), f"Expected shape ({B}, {T}), got {out.shape}"
