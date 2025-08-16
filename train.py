
from Patchouli.schemas.train_method import DEFAULT
from Patchouli.engine.trainer import Trainer
from Patchouli.brain import PatchouliTokenizer
from Patchouli.brain import PatchouliModel

if __name__ == "__main__":
    t = Trainer(
        cfg=DEFAULT,
        model=PatchouliModel(),
        tokenizer=PatchouliTokenizer(),
        brain_validator=None
    )

    t.start()