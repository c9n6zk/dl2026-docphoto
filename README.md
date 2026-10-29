# Lab 3 — a baseline and a first network

This folder is the project template with the plumbing filled in: a data
loader, a resize cache, a pixel table, seven handcrafted features, a
gradient-boosting baseline, Weights & Biases logging and the skeleton of a
training script. Six places are left for you, marked `TASK` in the source:
the grouped split, an MLP on raw pixels, the optimiser, the minibatch loop,
prediction and validation. The notebook `notebooks/lab3_baseline.ipynb` walks
through them in 90 minutes.

By the end of the lab this folder is your project repository, public on
GitHub, with the lab solved and committed. The project description has the
rules; the rest of this README has the infrastructure.

## Before the lab

Do these at home. In class they cost the hour.

**1. Docker and the environment.**

```bash
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
docker compose build
docker compose run --rm dev python scripts/check_env.py
```

If your machine has an NVIDIA GPU, uncomment the `COMPOSE_FILE` line for your
operating system in `.env`. Today's lab runs fine on a CPU.

**2. Weights & Biases.** Sign up or log in at https://wandb.ai, then copy your
API key from **https://wandb.ai/authorize** and paste it into `.env` after
`WANDB_API_KEY=`. Without a key your runs go to an anonymous account; that
works for today, not for the semester. Never commit `.env`.

**3. The training data.** It is a private Kaggle dataset, about 800 MB. The
link is on the lab slides and on Moodle. **Do not share the link and do not
re-upload the photographs anywhere**: they are your classmates' photographs and
for this course only.

Log in to Kaggle with the one account you will use for the whole semester,
open the link, press **Download**, and unpack the ZIP into this repository so
that `data/raw/train.csv` and `data/raw/train/` (1841 `.jpg` files) exist:

```bash
unzip ~/Downloads/archive.zip -d data/raw                                    # Linux, macOS
```

```powershell
Expand-Archive "$HOME\Downloads\archive.zip" -DestinationPath data\raw      # Windows PowerShell
```

The ZIP may be saved under another name; use whatever your browser saved.

**4. The caches and the tests.**

```bash
docker compose run --rm dev python -m src.data.dataset      # must report 1841 images; a few minutes
docker compose run --rm dev pytest                          # the two split tests fail: they are TASK 1
```

If the first command reports a number other than 1841, or cannot find
`train.csv`, the files are one folder too deep or too shallow:
`data/raw/train.csv` has to exist exactly there.

## Running the notebook from Docker

Nothing runs on your host except Docker. Jupyter runs inside the container
and you open it in a browser.

1. From the repository root, start Jupyter Lab:

   ```bash
   docker compose up lab
   ```

2. Wait for the line `Jupyter Server ... is running at` and open
   **http://localhost:8888** in your browser. There is no password or token.
3. In the file browser on the left, open `notebooks/lab3_baseline.ipynb`.
4. Edit the `src/` files in your own editor on the host. The project folder
   is mounted into the container, so the notebook sees every saved change;
   rerun the cell, and its `importlib.reload` picks the change up.
5. To stop, press `Ctrl+C` in the terminal, then run `docker compose down`.

**From VS Code instead of the browser.** Start the container as in step 1,
open the notebook in VS Code, click **Select Kernel → Existing Jupyter
Server**, and enter `http://localhost:8888`.

**If something goes wrong.**

| symptom | fix |
|---|---|
| `port is already allocated` | another Jupyter is running: `docker compose down`, or stop the other one |
| the notebook does not see your W&B key | you edited `.env` after starting: `docker compose down`, then `docker compose up lab` |
| `ModuleNotFoundError: src` | the notebook was opened from outside the container; use the browser at localhost:8888 |
| a cell keeps running old code | save the file, then rerun the cell; if it persists, *Kernel → Restart* |

Git runs on your host, not in the container: commit and push from a host
terminal in this folder.

---

## Everyday commands

```bash
docker compose run --rm dev bash                    # interactive shell
docker compose run --rm dev pytest                  # tests
docker compose run --rm dev ruff check .            # lint
docker compose run --rm dev python -m src.train --config configs/cnn.yaml
docker compose up lab                               # Jupyter Lab, localhost:8888
```

There is deliberately no Makefile. The commands above *are* the interface, and
when something breaks you will be debugging Docker rather than a wrapper around
it.

## How the container is put together

Three details, because they are the ones that break when you change something.

**The project is mounted, not copied.** Editing a file on your host changes it
in the container immediately. You only need `docker compose build` again when
`requirements.txt` or the `Dockerfile` changes.

**Torch comes from the base image and is deliberately not in
`requirements.txt`.** Installing it again would fetch a second, possibly
CPU-only build and quietly waste your GPU.

**The container runs as a normal user, not root**, so the files it writes into
your project are yours. If your host user id is not 1000 you will see permission
errors; rebuild with `docker compose build --build-arg UID=$(id -u)`.

---

## Layout

```
src/
  config.py            project root, target column, seed
  data/
    dataset.py         the table, the 768 px cache, the pixel table
    splits.py          the grouped split (TASK 1)
  features/
    handcrafted.py     classical image quality features
  models/
    baseline.py        features + a classical regressor
    mlp.py             the first network (TASK 2)
    cnn.py             convolutional models, a stub for later
  seed.py              reproducibility
  tracking.py          Weights & Biases
  train.py             the training script (TASKS 3-6 inside)
  evaluate.py          scoring and error breakdown
  predict.py           writes a submission file  <- read the contract in it

configs/               one YAML per experiment; lab3_mlp.yaml is the in-class run
scripts/check_env.py   verifies the container, prints versions and GPU status
notebooks/             exploration; move anything reusable into src/
tests/                 start with test_splits.py
data/raw/              what you were given, read only
data/work/             everything derived from it, regenerable
models/                checkpoints
submissions/           competition files
```

Nothing under `data/`, `models/` or `submissions/` is tracked by git.

## My results

Everything below is measured on strict unit 4 (380 scoreable photographs of
receipts and tables), the validation unit that `build_splits` holds out with
split seed 3. Lower is better.

| model | val RMSE | note |
|---|---|---|
| constant (training mean) | 0.2362 | the yardstick |
| gradient boosting, 7 handcrafted features | 0.1732 | the baseline to beat |
| MLP on 64 x 64 pixels, last of 60 epochs | 0.2512 | lab 3, `configs/lab3_mlp.yaml`; worse than the constant |
| MLP, best epoch picked **on the validation rows** | 0.2243 | epoch 2; optimistic by construction, shown only to measure the bias |
| MLP, early stopping on held-out **training** units | **0.2356** | `configs/lab3_mlp_es.yaml`, epoch 6, untouched validation unit |
| gradient boosting, 7 features + `ink_fraction` | 0.1719 | my feature; 0.0013 better, within the noise |

**Honest early stopping.** `train_mlp` sets aside whole strict units of the
training rows (`train.early_stopping` in the config: unit 6, 284 rows, chosen
by a seed that keeps the two-column family in training through unit 5), keeps
the weights of the epoch that scores best on them, and only then reports the
validation unit. The validation rows decide nothing. `tests/test_early_stopping.py`
asserts that the three sets of units are disjoint and that the reported score
is the one of the kept epoch.

What it shows: picking the best epoch on the validation rows themselves
flatters the MLP by about 0.011 (0.2243 against 0.2356). Honestly chosen, it
lands level with the constant predictor, which is what the curves said: the
training RMSE falls to 0.09 while the validation RMSE never leaves the
constant's neighbourhood. One validation unit and one early-stopping unit are
one sample each, so differences of a few hundredths are not evidence either
way.

**`src/predict.py` for the MLP checkpoint.** `load_model` rebuilds the network
from the checkpoint's own config and standardiser, `preprocess` turns one file
into pixels, `predict_one` returns one number in [0, 1]. Checked on the
checkpoint of the early-stopping run: twenty training photographs differ from
`train.predict` by at most 1e-7, and the whole validation unit scores 0.2356
through `predict.py` and through the training loop alike. One trap found on
the way: the training pixels are made from the 768 px cache (resize, JPEG,
read back), not from the original, and going straight from a 2048 px original
to 64 px changes single grey levels by up to 0.43. `preprocess` therefore
takes the cache's road and reproduces the training table bit for bit.
`tests/test_predict.py` pins this.

**A feature of my own: `ink_fraction`.** I did not look at the validation unit
to design it. The baseline's worst errors were found out-of-fold on the five
*training* units (hold out each in turn; 0.2038 RMSE overall), and the worst
photographs looked at in the native 2048 px originals. Two kinds stood out:

* photographer P142's `sparse` pages, label about 1.0 (perfectly readable),
  predicted about 0.3. The text is sharp but small and grey, and the page
  fills a tenth of a tall 946 x 2048 frame; the rest is white table and
  trousers. Sharpness over the whole frame, or over its *blurriest* inner
  tile, is dominated by what is not text.
* a motion-blurred sparse page and a grainy receipt predicted as sharp (label
  0.12 and 0.0, predicted 0.84 and 0.93). I suspect the grain: it has a large
  Laplacian variance of its own. I did not test that.

What the seven miss is *where the text is*. They measure paper, table and
grain exactly as they measure letters. I tried four ways of looking at the ink
only (a black-hat transform picks out dark strokes thinner than 11 px at
768 px), each added to the seven and scored out-of-fold on the training units:

| added feature | what it measures | out-of-fold RMSE |
|---|---|---|
| none (the seven) | | 0.2038 |
| `ink_edge` | edge gradient over stroke contrast on the ink, the sharpness of the letters themselves | 0.2035 |
| `ink_lap` | the same with the Laplacian | 0.2058 |
| `ink_contrast` | median stroke contrast | 0.2039 |
| **`ink_fraction`** | share of the frame covered by ink | **0.1963** |

The measures of how sharp the letters are did not help; the *amount of text in
the frame* did, and that is the one I kept (`src/features/handcrafted.py`,
`tests/test_features.py`). On the untouched validation unit the baseline goes
from 0.1732 to 0.1719.

Read that with care. It is a small gain, not an achievement: 0.0013 is far
inside the run-to-run and unit-to-unit noise, and I picked it as the best of
four on only five training units. The worst photographs did not get fixed
either: the P142 pages move from 0.30 to about 0.4 against a label of 1.0, and
P142's RMSE goes from 0.44 to 0.40. A model that has never seen a `sparse`
page, because unit 1 is the only unit that holds that family and it is the one
held out, cannot learn what a good one looks like from eight numbers. That,
not the features, is probably the limit here.

## Lab 4: a CNN from scratch

Same validation unit as above (strict unit 4, 380 scoreable photographs), RMSE,
lower is better. All of these are one run on one seed unless a row says
otherwise; the seed-to-seed spread of this network is 0.005 to 0.007 (below),
so differences under about 0.01 are not results.

| model | val RMSE | note |
|---|---|---|
| constant (training mean) | 0.2362 | |
| gradient boosting, handcrafted features | 0.1719 | now eight features: lab 3's `ink_fraction` is in `FEATURES`, the lab 3 seven gave 0.1732 |
| SmallCNN, 256 px, 8 epochs | 0.2107 | 24,761 parameters, 8 s per epoch on the CPU |
| SmallCNN, 768 px (stride-4 stem), 8 epochs | **0.1784** | 24,953 parameters; 768 minus 256: -0.0324, 95 % over pages [-0.0459, -0.0175] |
| + flip and shift, 8 epochs | 0.1816 | +0.0033 [-0.0033, +0.0090] against no augmentation: a tie |
| + flip and shift, 12 epochs | 0.1697 | -0.0087 [-0.0137, -0.0043] against the 8 epoch run |
| + flip and shift, 16 epochs | 0.1692 | -0.0092 [-0.0178, -0.0007] |
| + flip and shift, 16 epochs, honest early stopping | **0.1618** | epoch 15, chosen on a held-out *training* unit; see below |

**Resolution is what moves the number.** At 64 px a 9 pt letter is 0.7 px tall
(lab 3's MLP ended at 0.2512, worse than the constant); the same kind of network
at 768 px reaches the handcrafted baseline with forty times fewer parameters than the MLP.
The stem reads all 590,000 pixels but writes a 192 x 192 map, so nine times the
pixels cost about three times the time per epoch (8 s to 24 s on the CPU).

**The other rungs (768 px, 8 epochs, one run each).** Shuffled labels 0.2341
(the check passes: it stays at the constant, 0.2362; on permuted targets the
training RMSE does not leave it either, 0.2334, so in 8 epochs this small network cannot even memorise); weight decay 0: 0.1783; weight decay
0.5: 0.1754; GroupNorm: 0.1725; `resnet_small` (123,289 parameters): 0.1897 with
a training RMSE of 0.1751, the only model whose training score runs ahead of
its validation score. Keeping the best epoch *as chosen on the validation set*
gave 0.1739, 0.0045 better than the same run's last epoch: that is the bias of
peeking, and no model change.

**Honest early stopping for the CNN.** `train.early_stopping` in the config
(`configs/lab4_768_16ep_safe_es.yaml`) sets strict unit 6 aside from the
training rows, trains on the other 1,177, and keeps the epoch with the best score on
unit 6 (epoch 15, 0.1703 there). The validation unit then reports 0.1618, and
it chose nothing. Its own curve bottomed at 0.1553 in epoch 13, so picking
the epoch on validation would have reported a number 0.0065 too good. Two
cautions: this is one run, and it is 0.0165 better than the 8 epoch reference but only about 0.008
better than the 12 and 16 epoch runs, which is about one seed spread, and it trains on fewer rows, so
"early stopping helps" is not what this table shows; "it reports an honest number" is.

**Does it overfit, and where?** (homework 2) Not in 16 epochs. The validation RMSE
falls from 0.2121 to 0.1692, the training RMSE (on augmented images) from 0.2273 to
0.1759 and stays *above* it. The validation curve is noisy from epoch to
epoch (0.1645 and 0.2215 at epochs 9 and 8 of the 12 epoch run), which is another reason to
choose an epoch on held-out training rows and not by eye.

**Homework question 5: the network does not overfit; should the next experiment
change regularisation or capacity?** Capacity, or training time. The evidence, with
numbers: the training RMSE (0.176-0.178 at 12 and 16 epochs) stays above the
validation RMSE (0.169-0.170), which is underfitting; four more epochs bought
-0.009 where weight decay 0 and 0.5 moved the score by 0.0001 and 0.003, inside
the noise; and the one model with more capacity, `resnet_small`, is the only
one that begins to overfit (training 0.1751 against validation 0.1897, a gap of
0.0146). Regularisation comes after the curves separate. Next: more epochs and a wider
network or a pretrained backbone, one change per run. Caveat: `resnet_small`
scored worse than the SmallCNN here, in one run, so more capacity is not
automatically better at this budget.

**Run-to-run noise.** Three seeds of the 768 px run on the Mac GPU: no
augmentation 0.1732, 0.1795, 0.1726 (mean 0.1751, spread 0.0069); with flip and
shift 0.1787, 0.1774, 0.1735 (mean 0.1765, spread 0.0052). The same seed on the
CPU gave 0.1784 against 0.1732 on the GPU: a change of device moves the score
as much as a change of seed, which is why every comparison above is read
against about 0.005.

**Device.** `train_cnn.default_device()` picks CUDA, then the Mac GPU (MPS), then the
CPU; `DL_DEVICE=cpu` forces one. On the 768 px run, 8 epochs: 3.5 s per epoch on
MPS against 24 s on the CPU. The two checks before training (overfit one batch,
the same seed twice) pass on MPS too.

**`predict.py` for a CNN checkpoint** (homework 1). The dry run on the
validation unit through `src/predict.py` reproduces the training RMSE to five
decimals: 0.17835 at 768 px and 0.21071 at 256 px. It did not at first
(0.1782 against 0.1784). The training table stores photographs as bytes, rounded to
a whole grey level, and `inputs.preprocess_one` had skipped the rounding; half a grey
level was enough for the 768 px network to show it in the fourth decimal. Rounding
is now in `preprocess_one`, and `tests/test_predict.py` pins it.

## Making a Kaggle submission

One command produces one file:

```bash
docker compose run --rm dev python -m src.predict \
    --config configs/cnn.yaml \
    --checkpoint models/cnn_best.pt \
    --out submissions/cnn_best.csv
```

Then upload `submissions/cnn_best.csv` to the competition page.

The file has exactly two columns, `image_id` and `usability`, one row per test
image in the order of the competition's `sample_submission.csv`, every value a
float in [0, 1]. `src/predict.py` builds the output *from the sample
submission* rather than from a directory listing, so the ids and the row order
cannot drift, and it asserts the row count, uniqueness, no NaNs and the value
range before writing. Each of those has cost somebody a submission slot.

**One image in, one prediction out.** Your prediction for an image depends on
that image and your trained weights only, which is why the loop in `predict.py`
handles one image at a time and shares nothing between iterations. Batching for
speed is fine. Pooling anything across the test set is not; section 2 of the
project description says why.

**The checkpoint is part of the submission.** You hand in the weights that
produced your predictions, not a later or better version. Log every checkpoint
to W&B as an artifact of the run that made it and this question answers itself.

## The order of work

1. **Write the split.** Before any model. See below.
2. **Look at the data.** Plot the label distribution. Compare the best and
   worst photos *at full resolution*: a thumbnail hides the difference between
   a dark page and a blurred one.
3. **Train a network.** Pretrained backbone, regression head. Get one run
   working end to end, with a checkpoint and a submission file, before you
   improve anything.
4. **Compare it against the baseline.** The instructor publishes a classical
   baseline on the leaderboard, built from handcrafted image features, and your
   network has to beat it. Building your own version takes an afternoon and
   shows you what a whole-image statistic can and cannot see.
5. **Improve it.** Resolution first, then augmentation, architecture,
   ensembling. Change one thing per run.

## Write the split first

The same page appears in dozens of photographs, every layout exists in a
Hungarian and an English version typeset identically, and each photographer's
photos share a phone and a room. Split rows at random and near-identical photos
land in training and validation at the same time, and the score you report is
optimistic.

We measured it on this data with a ResNet-18 at 768 px, scoring the same model
four ways:

| validation scheme | RMSE |
|---|---|
| random 5-fold | 0.113 |
| folds by `group` | 0.136 |
| folds by `group_strict` | 0.143 |
| the held-out test set | **0.148** |

The random split is a fifth too good. `group`, in which no page and no
photographer appears twice, is still 8 % too good: the two language versions
of a layout sit in two neighbouring groups, and a network recognises the
layout. `group_strict` keeps both versions together, which is how the test set
was cut, and it lands on the test. The stronger the model, the more it profits
from every leak, so the mistake grows with your progress.

**Split by `group_strict`.** The training set has six strict units of ten pages
each, and each covers one or two document families, so which unit you hold out
matters: two of them are the only source of their family. The split therefore
has its own seed, separate from the model's, so that rerunning a model with
another seed does not move the validation photographs. `tests/test_splits.py`
asserts that no unit appears on both sides. It is a few lines, and it protects
every number you report this semester.

Implementing the random split too and measuring your own gap makes a far more
convincing paragraph in your report than a number we gave you.

## Things that will cost you if you get them wrong

**Augmentation that changes image quality.** Flips and small translations are
safe. Blur, brightness jitter and added noise change the very thing you are
predicting, so they corrupt the label. This is the most common way to make a
model quietly worse.

**Metadata as model input.** `page_id`, `family`, `font`, `size_pt`,
`photographer` and the rest exist only in `train.csv`; a test image is an image
and nothing else. Use the columns to split and to slice your error, never as
features.

**Measuring on `eval_ok = 0`.** Those 47 photographs were uploaded heavily
downscaled. Train on them if you like, but leave them out of every number you
report.

**Too little resolution.** At 384 px a ResNet-18 loses to the handcrafted
baseline; the defects that matter are a few pixels wide on a 2048 px frame.
Resize once into `data/work/` and start at 768.

**Reporting a single number.** RMSE tells you how well you did, not what to fix.
Break the error down by document family, language, font, font size and
photographer.

**Not fixing the seed.** You will compare dozens of runs. A difference smaller
than your run-to-run variance is not a result.

**Committing data or checkpoints.** `.gitignore` covers it, but check
`git status` before your first commit rather than after.

## Notes on the data

The photographs have their long side at 2048 px and no metadata. The fiducial
markers the labelling relied on have been painted out, so the four flat patches
near the page corners are not part of the document, and you cannot rectify the
page from them. The page edges are still there if you want to try.

The defects that matter most are the **local** ones: a smear across one column,
a crease through a paragraph. The same defect over text and over empty margin
looks nearly identical to a whole-image statistic and produces very different
labels. That is the core difficulty of this task, and it is why an average of
sharpness or brightness over the frame only gets you so far.

`train_annotations.csv` and `train_consensus.csv` hold human ratings of about a
third of the photographs, on a 1 to 5 scale, from several raters each. The
competition does not score them. Where people and the OCR engine disagree, and
whether the disagreement is structured, is an open question on data you helped
create.
