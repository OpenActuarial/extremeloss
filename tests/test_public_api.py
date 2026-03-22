from __future__ import annotations

import extremeloss


def test_public_api_exports_expected_symbols():
    assert hasattr(extremeloss, 'fit_gpd')
    assert hasattr(extremeloss, 'estimate_tail_probability')
    assert hasattr(extremeloss, 'extreme_loss_summary')
    assert extremeloss.__version__ == '0.1.0'
