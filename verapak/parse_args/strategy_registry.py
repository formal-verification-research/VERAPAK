from verapak import verification, partitioning, abstraction, dimension_ranking


VERIFICATION_STRATEGIES = verification.STRATEGIES

PARTITIONING_STRATEGIES = partitioning.STRATEGIES

DIMENSION_RANKING_STRATEGIES = dimension_ranking.STRATEGIES

ABSTRACTION_STRATEGIES = abstraction.STRATEGIES

ALL_STRATEGIES = {
    'verification': VERIFICATION_STRATEGIES,
    'partitioning': PARTITIONING_STRATEGIES,
    'dimension_ranking': DIMENSION_RANKING_STRATEGIES,
    'abstraction': ABSTRACTION_STRATEGIES}
