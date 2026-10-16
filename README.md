# Foundations of Vision and Language

Course workspace for tracking an M.Tech subject over time. The repository is
organized by class so that notes, notebooks, implementations, and tests stay
together and remain easy to revisit.

## Repository layout

```text
.
├── classes/
│   ├── README.md
│   └── class_NN_short-topic/
│       ├── README.md
│       ├── notes/
│       ├── notebooks/
│       ├── src/
│       └── tests/
├── README.md
├── pyproject.toml
└── uv.lock
```

See [`classes/README.md`](./classes/README.md) for the convention used when
adding a new class. The first class currently available is
[Foundations of neural networks](./classes/class_01_foundations_of_neural_networks/).

## How to add a class

1. Create the next numbered directory:
   `classes/class_02_<short-topic>/`.
2. Add a class README with learning goals, materials, status, and follow-up
   ideas.
3. Store source material in `notes/`, exploratory work in `notebooks/`, and
   reusable code in `src/`.
4. Add tests under `tests/` for code that should remain reliable.
5. Update the class list in this README.

## Development with uv

[`uv`](https://docs.astral.sh/uv/) manages the project environment,
dependencies, and lockfile. Install uv once if needed:

For me I used below command because I code on macos
```bash
brew install uv
```

From the repository root, create the environment and install the locked
dependencies:

```bash
uv sync
```

Run the complete test suite:

```bash
uv run pytest
```

`uv run` automatically uses the project environment. Commit `uv.lock` so the
same dependency versions can be restored later with `uv sync`.

## Running a given class

Commands are run from the repository root. Replace the class path with the
class you want to study.

Run every test for Class 01:

```bash
uv run pytest classes/class_01_foundations_of_neural_networks/tests -q
```

Open the Class 01 notebook:

```bash
uv run jupyter notebook \
  classes/class_01_foundations_of_neural_networks/notebooks/flowers.ipynb
```

You can also start JupyterLab:

```bash
uv run jupyter lab
```

Then open the notebook under
`classes/class_01_foundations_of_neural_networks/notebooks/`.

Run a reusable Python module from a class:

```bash
uv run python -c "
from classes.class_01_foundations_of_neural_networks.src.flower_classifier import (
    DEFAULT_DATA,
    FlowerClassifier,
)

model = FlowerClassifier(seed=7)
model.fit(DEFAULT_DATA[:, :2], DEFAULT_DATA[:, 2])
print(model.evaluate(DEFAULT_DATA[:, :2], DEFAULT_DATA[:, 2]))
"
```

When a class gains a standalone script, keep it in that class's `src/`
directory and run it with `uv run python path/to/script.py`.

The course is primarily a learning record: notebooks may be exploratory,
while `src/` contains the parts worth reusing and testing.
