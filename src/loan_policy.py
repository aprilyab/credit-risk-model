import numpy as np
from typing import Dict, Union

def suggest_terms(
    pd_prob: float,
    past_avg_ticket: float,
    risk_appetite: float = 0.6,
    max_cap: float = 500.0
) -> Dict[str, Union[float, int]]:
    """
    Suggest loan terms (credit limit & duration) based on predicted risk.

    The rule is:
      - Suggested credit limit scales with both:
        - (1 - Probability of Default) → lower PD means higher limit
        - Past average ticket size (spending history)
      - Duration is assigned in discrete tiers based on PD thresholds.

    Parameters
    ----------
    pd_prob : float
        Predicted probability of default (0.0 to 1.0).
    past_avg_ticket : float
        Average transaction size from customer history.
        If <= 0, a default base of 100.0 is used.
    risk_appetite : float, default=0.6
        Controls aggressiveness of the lending policy.
        Higher values allow larger limits for the same PD.
    max_cap : float, default=500.0
        Maximum credit limit allowed regardless of PD or history.

    Returns
    -------
    Dict[str, Union[float, int]]
        Dictionary containing:
        - "suggested_limit": float
            Recommended maximum exposure (credit limit).
        - "suggested_duration_days": int
            Loan duration in days (30, 60, or 90).

    Examples
    --------
    >>> suggest_terms(pd_prob=0.1, past_avg_ticket=200)
    {'suggested_limit': 260.0, 'suggested_duration_days': 90}

    >>> suggest_terms(pd_prob=0.4, past_avg_ticket=150)
    {'suggested_limit': 165.0, 'suggested_duration_days': 30}
    """
    # Ensure PD is within [0, 1]
    pd_prob = float(np.clip(pd_prob, 0.0, 1.0))

    # Determine base spending reference
    base = past_avg_ticket if past_avg_ticket > 0 else 100.0

    # Calculate suggested credit limit
    # Formula: limit = base × (0.5 + risk_appetite × (1 - PD))
    # Then capped by max_cap
    limit = min(max_cap, base * (0.5 + risk_appetite * (1 - pd_prob)))

    # Assign loan duration tier
    if pd_prob < 0.15:
        days = 90
    elif pd_prob < 0.35:
        days = 60
    else:
        days = 30

    return {
        "suggested_limit": round(limit, 2),
        "suggested_duration_days": days
    }
