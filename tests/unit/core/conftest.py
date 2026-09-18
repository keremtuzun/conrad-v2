from __future__ import annotations

import numpy as np
import pytest
import torch

from conrad.core.config import tiny_config
from conrad.core.pipeline import Model2Core
from conrad.core.pmbl import PMBL
from conrad.evaluation.core_experiments.evidence_factory import EvidenceFactory
from conrad.persistence.db import make_engine, migrate
from conrad.persistence.repository import Repository
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Domain


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def cfg():
    return tiny_config()


@pytest.fixture
def ids():
    return IdFactory(1234)


@pytest.fixture
def fac(ids, cfg):
    return EvidenceFactory(ids, cfg.evidence_dim, np.random.default_rng(0))


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "core.db"
    migrate(path)
    return Repository(make_engine(path))


@pytest.fixture
def pmbl(repo, fac, cfg, ids):
    return PMBL(repo, fac.run_id, cfg, ids)


@pytest.fixture
def core(repo, fac, cfg, ids):
    return Model2Core(cfg, repo, fac.run_id, ids, Domain.TECHNICAL, "component")


def _archived(pmbl, fac, *args, **kwargs):
    ev, rec = fac.make(*args, **kwargs)
    pmbl.archive_evidence(ev, [rec])
    return ev


@pytest.fixture
def archived():
    """archived(pmbl, fac, time_s, measurements, **kw) -> Evidence already in the archive."""
    return _archived
