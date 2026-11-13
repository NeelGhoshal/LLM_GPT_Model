# TinyStories GPT

A GPT language model written and trained from scratch on a single laptop GPU
(RTX 4060, 8 GB). No pretrained weights and no pretrained tokenizer: the
tokenizer, the model and the training loop are all in this repo.

It is trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
about 2.1 million short stories written in the vocabulary of a young child.
The narrow domain is what lets a model this small learn fluent English in a
few hours.

## Files

| File | What it does |
|---|---|
| `prepare_data.py` | Downloads TinyStories, trains an 8,192-token BPE tokenizer, encodes the stories to `train.bin` / `val.bin` |
| `model.py` | The GPT: 8 layers, 8 heads, width 512, context 512 tokens, 29.6M parameters |
| `train.py` | Training loop: bf16, AdamW, warmup and cosine decay, wall-clock budget, checkpoints, `--resume` |
| `sample.py` | Generates stories from the trained weights |
| `tokenizer.json` | The trained tokenizer |

## Run it

```
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python prepare_data.py
python train.py --max-hours 6
python sample.py "Once upon a time, a little fox"
```

`train.py --resume` continues an interrupted run from the last checkpoint.
Use `--data-dir` and `--out` to keep the token files and checkpoints (over
1 GB together) outside a synced folder.

## Results

Trained for 6 hours on the RTX 4060 laptop GPU: 88,150 steps of 32 x 512
tokens, 1.44 billion tokens in total, about three passes over the data.

| | |
|---|---|
| Validation loss | 1.285 (perplexity 3.6) |
| Training loss at the end | about 1.25 |
| Speed | 66k to 75k tokens per second |
| VRAM used | 4.5 GB |

Validation loss is measured on a fixed sample of 655k tokens from the
TinyStories validation split. It was still falling slowly when the time
budget ran out (1.296 at step 78,000, 1.285 at the end).

Validation loss during training:

| Step | Hours | Validation loss |
|---|---|---|
| 2,000 | 0.14 | 2.002 |
| 10,000 | 0.70 | 1.592 |
| 20,000 | 1.39 | 1.504 |
| 40,000 | 2.90 | 1.413 |
| 60,000 | 4.14 | 1.348 |
| 80,000 | 5.44 | 1.293 |
| 88,150 | 6.00 | 1.285 |

### Samples

Prompts are in bold; the rest is generated (temperature 0.8, top-k 50).

> **The little dragon was afraid of** the dark. He didn't want to get out of
> his cave, he wanted to be alone.
>
> One day, the dragon heard a loud noise coming from outside. He started to
> panic and his heart was pounding. He ran to find his mother.
>
> "Mama! The monster is coming!" he cried.
>
> His mother smiled and said, "It's alright, my little one. The monster is
> just a harmless creature. He won't hurt you."
>
> The dragon was still scared, but he asked his mother, "Will the monster
> come back?"
>
> His mother replied, "No, everything will be ok. The monster won't come
> back. He's just looking for some friends."
>
> The dragon was feeling better with his mother's words. He gathered some of
> his courage and told his mother he was ready to face the monster. His
> mother smiled and said, "That's my brave little dragon."

> **Tom found a red ball in the garden.** He wanted to play with it. He threw
> the ball high and far. He caught it and threw it again. But he threw it too
> hard. The ball flew over the fence and landed in the road. [...]

## Limits

- The model only knows the world of TinyStories. It writes simple stories
  and nothing else: no facts, no instructions, no questions and answers.
- Stories are locally coherent, but plots can wander or contradict
  themselves over a few paragraphs.
- Some stories in TinyStories are sad or end badly, and the model reproduces
  that.
- About 6% of TinyStories stories have broken quote characters (`â€œ` in
  place of a curly quote). The data is used as published, so the model
  sometimes writes them too.
