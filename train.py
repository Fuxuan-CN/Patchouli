
from Patchouli.schemas.train_method import DEFAULT
from Patchouli.train_and_gen.trainer import Trainer
from Patchouli.brain import PatchouliTokenizer
from Patchouli.brain import PatchouliModel
from Patchouli.utils.exec_hook import set_exechook

set_exechook()

if __name__ == "__main__":
    t = Trainer(
        cfg=DEFAULT,
        model=PatchouliModel(),
        tokenizer=PatchouliTokenizer(),
        brain_validator=None
    )

    t.start()