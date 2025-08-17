import numpy as np

class Scorecard:
    """
    Logistic scorecard mapping: converts predicted probabilities into a credit risk score
    using the "Points to Double Odds" (PDO) system.

    The score is calculated so that:
      - At a given base probability `p_at_base`, the score is `base_score`.
      - Increasing the odds by a factor of 2 increases the score by `pdo` points.

    Example:
        If `base_score` = 600 at odds 1:1 (p = 0.5), and PDO = 50:
        - A probability corresponding to odds 2:1 will have score = 650.
        - A probability corresponding to odds 4:1 will have score = 700.

    Parameters
    ----------
    base_score : float, default=600.0
        The score assigned to the base odds.
    p_at_base : float, default=0.5
        The probability at the base score.
    pdo : float, default=50.0
        Points to Double the Odds — the number of points by which the score changes
        when the odds double or halve.

    Attributes
    ----------
    base_score : float
        The score at base odds.
    odds0 : float
        The base odds corresponding to `p_at_base`.
    factor : float
        Scaling factor for log-odds transformation, calculated as `pdo / log(2)`.
    """

    def __init__(self, base_score: float = 600.0, p_at_base: float = 0.5, pdo: float = 50.0):
        self.base_score = base_score
        self.odds0 = p_at_base / (1 - p_at_base)
        self.factor = pdo / np.log(2)

    def prob_to_score(self, p: float) -> float:
        """
        Convert a probability into a credit score.

        Parameters
        ----------
        p : float
            Predicted probability of default (or event of interest). Must be between 0 and 1.

        Returns
        -------
        float
            The corresponding score based on the scorecard mapping.
        """
        # Clip probability to avoid divide-by-zero and log-of-zero errors
        p = np.clip(p, 1e-6, 1 - 1e-6)

        # Convert probability to odds
        odds = (1 - p) / p

        # Compute score using the logistic scorecard formula
        return float(self.base_score + self.factor * np.log(odds / self.odds0))
