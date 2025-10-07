"""
Attack Spanish Recipe
=====================

(Contextualized Perturbation for Spanish NLP Adversarial Attack)

"""

from content_moderation import Attack
from content_moderation.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from content_moderation.goal_functions import UntargetedClassification
from content_moderation.search_methods import GreedyWordSwapWIR
from content_moderation.transformations import (
    CompositeTransformation,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapWordNet,
)

from .attack_recipe import AttackRecipe


class SpanishRecipe(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        transformation = CompositeTransformation(
            [
                WordSwapWordNet(language="esp"),
                WordSwapChangeLocation(language="esp"),
                WordSwapChangeName(language="esp"),
            ]
        )
        constraints = [RepeatModification(), StopwordModification("spanish")]
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR()
        return Attack(goal_function, constraints, transformation, search_method)
