import sys
import argparse
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from content_moderation import Attacker, AttackArgs, DatasetArgs, CommandLineAttackArgs, ModelArgs

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
def parse_args():
    parser = argparse.ArgumentParser(description="Run TextAttack-style attacks programmatically.")
    parser.add_argument("--model", type=str, default='distilbert-base-uncased-imdb', help="HuggingFace model.")
    parser.add_argument("--model-from-file", type=str, help="Local model.")
    
    parser.add_argument("--attack_recipe", type=str, default="textfooler", help="Attack recipe to run.")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to attack.")
    return parser.parse_args()
# "--model-from-file" "--model-from-huggingface"

def run_attack(args):
    attack_args = CommandLineAttackArgs(**vars(args))
    dataset = DatasetArgs._create_dataset_from_args(attack_args)
    if attack_args.interactive:
            model_wrapper = ModelArgs._create_model_from_args(attack_args)
            attack = CommandLineAttackArgs._create_attack_from_args(
                attack_args, model_wrapper
            )
            Attacker.attack_interactive(attack)
    else:
            model_wrapper = ModelArgs._create_model_from_args(attack_args)
            attack = CommandLineAttackArgs._create_attack_from_args(
                attack_args, model_wrapper
            )
            attacker = Attacker(attack, dataset, attack_args)
            attacker.attack_dataset()
    

if __name__ == "__main__":
    args = parse_args()
    run_attack(args)

