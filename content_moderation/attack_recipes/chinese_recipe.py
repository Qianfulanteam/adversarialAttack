"""
Attack Chinese Recipe
=====================

(Contextualized Perturbation for Chinese NLP Adversarial Attack)

"""

import string

from content_moderation import Attack
from content_moderation.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from content_moderation.goal_functions import UntargetedClassification
from content_moderation.search_methods import GreedyWordSwapWIR
from content_moderation.shared.data import CHN_STOPWORD
from content_moderation.transformations import (
    ChineseHomophoneCharacterSwap,
    ChineseMorphonymCharacterSwap,
    ChineseWordSwapHowNet,
    ChineseWordSwapMaskedLM,
    CompositeTransformation,
)

from .attack_recipe import AttackRecipe


class ChineseRecipe(AttackRecipe):
    """An implementation of the attack used in "Beyond Accuracy: Behavioral
    Testing of NLP models with CheckList", Ribeiro et al., 2020.

    This attack focuses on a number of attacks used in the Invariance Testing
    Method: Contraction, Extension, Changing Names, Number, Location

    https://arxiv.org/abs/2005.04118
    """

    @staticmethod
    def build(model_wrapper):
        transformation = CompositeTransformation(
            [
                ChineseWordSwapHowNet(),
                ChineseWordSwapMaskedLM(),
                ChineseMorphonymCharacterSwap(),
                ChineseHomophoneCharacterSwap(),
            ]
        )

        stopwords = CHN_STOPWORD.union(set(string.punctuation))

        # Need this constraint to prevent extend and contract modifying each others' changes and forming infinite loop
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

        # Untargeted attack & Greedy search with weighted saliency
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="weighted-saliency")

        return Attack(goal_function, constraints, transformation, search_method)
