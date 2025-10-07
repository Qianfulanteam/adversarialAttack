"""
CheckList:
=========================

(Beyond Accuracy: Behavioral Testing of NLP models with CheckList)

"""

from content_moderation import Attack
from content_moderation.constraints.pre_transformation import RepeatModification
from content_moderation.goal_functions import UntargetedClassification
from content_moderation.search_methods import GreedySearch
from content_moderation.transformations import (
    CompositeTransformation,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapChangeNumber,
    WordSwapContract,
    WordSwapExtend,
)

from .attack_recipe import AttackRecipe


class CheckList2020(AttackRecipe):
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
                WordSwapExtend(),
                WordSwapContract(),
                WordSwapChangeName(),
                WordSwapChangeNumber(),
                WordSwapChangeLocation(),
            ]
        )

        # Need this constraint to prevent extend and contract modifying each others' changes and forming infinite loop
        constraints = [RepeatModification()]

        # Untargeted attack & GreedySearch
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
