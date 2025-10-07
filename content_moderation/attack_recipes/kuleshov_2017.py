"""
Kuleshov2017
==============
(Adversarial Examples for Natural Language Classification Problems)

"""

from content_moderation import Attack
from content_moderation.constraints.grammaticality.language_models import GPT2
from content_moderation.constraints.overlap import MaxWordsPerturbed
from content_moderation.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from content_moderation.constraints.semantics.sentence_encoders import ThoughtVector
from content_moderation.goal_functions import UntargetedClassification
from content_moderation.search_methods import GreedySearch
from content_moderation.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class Kuleshov2017(AttackRecipe):
    """Kuleshov, V. et al.

    Generating Natural Language Adversarial Examples.

    https://openreview.net/pdf?id=r1QZ3zbAZ.
    """

    @staticmethod
    def build(model_wrapper):
        #
        # "Specifically, in all experiments, we used a target of τ = 0.7,
        # a neighborhood size of N = 15, and parameters λ_1 = 0.2 and δ = 0.5; we set
        # the syntactic bound to λ_2 = 2 nats for sentiment analysis"

        #
        # Word swap with top-15 counter-fitted embedding neighbors.
        #
        transformation = WordSwapEmbedding(max_candidates=15)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Maximum of 50% of words perturbed (δ in the paper).
        #
        constraints.append(MaxWordsPerturbed(max_percent=0.5))
        #
        # Maximum thought vector Euclidean distance of λ_1 = 0.2. (eq. 4)
        #
        constraints.append(ThoughtVector(threshold=0.2, metric="max_euclidean"))
        #
        #
        # Maximum language model log-probability difference of λ_2 = 2. (eq. 5)
        #
        constraints.append(GPT2(max_log_prob_diff=2.0))
        #
        # Goal is untargeted classification: reduce original probability score
        # to below τ = 0.7 (Algorithm 1).
        #
        goal_function = UntargetedClassification(model_wrapper, target_max_score=0.7)
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
