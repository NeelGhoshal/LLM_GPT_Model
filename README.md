# TinyStories GPT

A GPT language model written and trained from scratch on a single laptop GPU
(RTX 4060, 8 GB). No pretrained weights and no pretrained tokenizer: the
tokenizer, the model and the training loop are all in this repo.

It is trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
about 2.1 million short stories written in the vocabulary of a young child.
The narrow domain is what lets a model this small learn fluent English in a
few hours.

After 6 hours of training it reaches a validation loss of 1.285 (perplexity
3.6) and writes coherent short stories:

> **The little dragon was afraid of** the dark. He didn't want to get out of
> his cave, he wanted to be alone. One day, the dragon heard a loud noise
> coming from outside. He started to panic and his heart was pounding. He ran
> to find his mother. "Mama! The monster is coming!" he cried.

## Contents

- [Files](#files)
- [Run it](#run-it)
- [Data](#data)
- [Tokenizer](#tokenizer)
- [Model](#model)
- [Training](#training)
- [Results](#results)
- [Sampling](#sampling)
- [Limits](#limits)
- [Ideas for next steps](#ideas-for-next-steps)

## Files

| File | What it does |
|---|---|
| `prepare_data.py` | Downloads TinyStories, trains the BPE tokenizer, encodes the stories to `train.bin` / `val.bin` |
| `model.py` | The GPT, about 120 lines |
| `train.py` | Training loop: bf16, AdamW, warmup and cosine decay, wall-clock budget, checkpoints, `--resume` |
| `sample.py` | Generates stories from the trained weights |
| `tokenizer.json` | The trained tokenizer |
| `out/model.pt` | Trained weights (65 MB, half precision); not tracked by git |

## Run it

Needs Python 3.10+ and an NVIDIA GPU. Training uses 4.5 GB of VRAM.

```
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python prepare_data.py
python train.py --max-hours 6
python sample.py "Once upon a time, a little fox"
```

| Step | Time on an RTX 4060 laptop | Disk |
|---|---|---|
| `prepare_data.py` | about 10 minutes, mostly download | 0.9 GB of token files, plus the dataset in the Hugging Face cache |
| `train.py` | whatever `--max-hours` says | 356 MB checkpoint, 65 MB exported weights |
| `sample.py` | a few seconds per story | |

`train.py --resume` continues an interrupted run from the last checkpoint,
including the optimizer state and the time already used. Use `--data-dir` and
`--out` to keep the token files and checkpoints outside a synced folder such
as OneDrive.

## Data

| Split | Stories | Tokens |
|---|---|---|
| train | 2,119,719 | 466.8M |
| validation | 21,990 | 4.7M |

A story averages about 220 tokens. Each split is stored as one flat array of
16-bit token ids, with an end-of-text token after every story. Training reads
the array through a memory map, so the 0.9 GB file never has to fit in RAM.

A training example is a random 512-token window from that array. Windows
start anywhere, so most begin in the middle of one story and run into the
next. The target is the same window shifted one token to the left: at every
position the model predicts the token that follows.

## Tokenizer

A byte-level BPE tokenizer with 8,192 tokens, trained on the first 300,000
training stories. Token 0 is `<|endoftext|>`.

The vocabulary is small on purpose. TinyStories uses a limited set of words,
so 8,192 tokens are enough to keep common words whole, and the embedding
table stays at 4.2M parameters instead of the 25.7M that GPT-2's
50,257-token vocabulary would need at this width.

```
"Once upon a time, a little fox found a shiny key."
['Once', ' upon', ' a', ' time', ',', ' a', ' little', ' fox', ' found', ' a', ' shiny', ' key', '.']
```

## Model

A decoder-only transformer in the GPT-2 style.

| | |
|---|---|
| Layers | 8 |
| Attention heads | 8 (64 dimensions each) |
| Embedding width | 512 |
| MLP width | 2,048 |
| Context length | 512 tokens |
| Vocabulary | 8,192 |
| Parameters | 29.6M |

Where the parameters are:

| Part | Parameters |
|---|---|
| Token embedding (shared with the output layer) | 4.19M |
| Position embedding | 0.26M |
| 8 transformer blocks, 3.15M each | 25.18M |
| Total | 29.63M |

Design choices:

- **Pre-norm blocks.** LayerNorm is applied before attention and before the
  MLP, which trains more stably than the original post-norm layout.
- **Learned position embeddings**, one vector per position, added to the
  token embeddings.
- **Weight tying.** The output layer reuses the token embedding matrix.
- **No biases** in the linear layers or LayerNorms.
- **GELU** activation in the MLP.
- **Scaled initialisation.** Weights start from a normal distribution with
  standard deviation 0.02; the layers that write into the residual stream are
  scaled down by `1/sqrt(2 * n_layer)`.
- **No dropout.** With three passes over 467M tokens the model does not
  overfit: training and validation loss stay within 0.04 of each other.

Attention uses PyTorch's `scaled_dot_product_attention` with a causal mask,
so each position sees only itself and earlier positions.

## Training

| Setting | Value |
|---|---|
| Batch | 32 sequences x 512 tokens = 16,384 tokens per step |
| Optimizer | AdamW, betas (0.9, 0.95) |
| Weight decay | 0.1 on weight matrices, none on LayerNorm gains |
| Learning rate | 6e-4 peak, linear warmup over 1,000 steps, cosine decay to 6e-5 |
| Gradient clipping | norm 1.0 |
| Precision | bf16 autocast |
| Time budget | 6 hours |

**Time-based schedule.** The cosine decay follows whichever is further along:
the step count against `--max-steps`, or the elapsed time against
`--max-hours`. On this GPU the clock is always ahead, so the learning rate
reaches its minimum exactly when the time runs out, whatever the speed.

**Checkpoints.** Every 2,000 steps the script measures validation loss on a
fixed set of 40 batches (655k tokens), prints a sample story, saves a full
checkpoint for `--resume`, and exports the weights if validation improved.

**Batch size.** 32 was chosen by measurement. At batch 64 the run needed
8.5 GB, spilled out of the 8 GB of VRAM into shared memory, and ran at about
half the speed.

The 6 hours of training were spread over six sessions, because the run was
interrupted five times and continued with `--resume`. Each interruption lost
the few minutes of work since the last checkpoint.

## Results

88,150 steps, 1.44 billion tokens, about three passes over the data.

| | |
|---|---|
| Validation loss | 1.285 (perplexity 3.6) |
| Training loss at the end | about 1.25 |
| Speed | 66k to 75k tokens per second |
| VRAM used | 4.5 GB |

A model that guessed uniformly over the vocabulary would score ln(8192) =
9.01. A perplexity of 3.6 means the model is, on average, about as uncertain
as if it were choosing between 3.6 equally likely next tokens.

| Step | Hours | Tokens seen | Validation loss |
|---|---|---|---|
| 2,000 | 0.14 | 33M | 2.002 |
| 10,000 | 0.70 | 164M | 1.592 |
| 20,000 | 1.39 | 328M | 1.504 |
| 40,000 | 2.90 | 655M | 1.413 |
| 60,000 | 4.14 | 983M | 1.348 |
| 80,000 | 5.44 | 1,311M | 1.293 |
| 88,150 | 6.00 | 1,444M | 1.285 |

Most of the learning happens early: the training loss falls from 7.9 to 2.0
in the first 8 minutes, and the remaining 5 hours 50 minutes take it to
about 1.25. Validation loss was still falling slowly at the end (1.296 at
step 78,000), so a longer run would improve it a little.

What the samples looked like along the way:

| Step | Sample |
|---|---|
| 300 | "She loved to the world and her mommy. One day, it got to play with a friend, Lily's mom." |
| 2,000 | "She loved to play outside and explore the world around her. One day, she saw a shiny necklace on the ground." |
| 14,000 | "Max felt bad and said, 'I'm sorry, Lily. I didn't mean to scare you. Can I pet you?'" |
| 52,000 | "Mommy said, 'Yes, it's so high up in the sky. Let's wait and see if the bird will come down.'" |

Word order comes first, then grammar, then consistent characters, then
events that follow from each other. (The step-300 sample is from a short
test run, not the main one.)

### Full samples from the final model

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

## Sampling

```
python sample.py "The little dragon was afraid of" --n 3 --temperature 0.9
```

| Option | Default | Effect |
|---|---|---|
| `prompt` | `Once upon a time` | Text the story starts from |
| `--n` | 1 | Number of stories |
| `--max-new-tokens` | 300 | Length limit; generation also stops at the end-of-text token |
| `--temperature` | 0.8 | Lower is safer and more repetitive, higher is more varied and less coherent |
| `--top-k` | 50 | Sample only from the k most likely tokens |
| `--seed` | none | Set it to get the same story again |
| `--weights` | `out/model.pt` | Weights file to load |

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
- Validation loss is measured on a fixed 655k-token sample of the validation
  split, not all 4.7M tokens.

## Ideas for next steps

- Repair the broken quote characters in `prepare_data.py` and retrain.
- Train longer, or raise `--n-layer` and `--n-embd`, and compare the loss
  curves.
- Replace the learned position embeddings with rotary embeddings.
- Add a key-value cache to `generate` so sampling does not recompute the
  whole context for every token.
