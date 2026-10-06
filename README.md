# Braess Paradox with Langroid Driver Agents

This project implements a traffic simulation with Langroid agents and compares its results with the Braess paradox example from the Multi-Agent Systems course slides. It also contrasts deterministic route choice with decisions made by a local language model.

## What is included

- `Code/braess_langroid_simulation.py` — traffic network, Langroid driver agent, heuristic and LLM decision modes, and CSV output.
- `Braess_Langroid_Report.ipynb` — analysis notebook that compares the saved runs and creates report tables and figures.
- `results_langroid_tolls_heuristic/` — heuristic baseline, including toll scenarios.
- `results_langroid_7b_toll40_checked/` and `results_langroid_7b_toll40_raw/` — Qwen 2.5 7B comparison runs with and without deterministic choice checking.
- `report_outputs/` — generated comparison figures and KPI table.
- `results_langroid_7b_comparison.md` — summary of the 7B checked and raw results.

Course PDFs, the local Python virtual environment, and earlier 7B/14B/32B experiment folders are not included. The reference values are from the Braess paradox slides in the Moodle course (`Multiagenten`).

## Results at a glance

The slide reference for the original network is 65 minutes; adding the A-B connection without a toll raises the reference average to 82 minutes. In the saved Qwen 2.5 7B run, the checked mode reproduces those values and reaches 64.8876 minutes with a toll of 40. The raw mode shows the model's uncorrected route choices and does not reach the same equilibrium.

The checked and raw modes answer different questions: `llm_checked` records when a valid model choice is non-minimal and applies the lowest-cost option; `llm_raw` accepts any valid model choice as returned.

## Setup

Python 3.12 was used for the saved notebook run. From the repository root, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the simulation

The deterministic baseline does not need a language model:

```powershell
python Code/braess_langroid_simulation.py --agents 500 --vehicles 4000 --decision-mode heuristic --output-dir results_langroid_tolls_heuristic
```

For LLM modes, install and start [Ollama](https://ollama.com/), then pull the model:

```powershell
ollama pull qwen2.5:7b
```

Run the checked and raw toll-40 experiments:

```powershell
python Code/braess_langroid_simulation.py --agents 500 --vehicles 4000 --decision-mode llm_checked --model qwen2.5:7b --tolls 40 --output-dir results_langroid_7b_toll40_checked
python Code/braess_langroid_simulation.py --agents 500 --vehicles 4000 --decision-mode llm_raw --model qwen2.5:7b --tolls 40 --output-dir results_langroid_7b_toll40_raw
```

The model endpoint defaults to `http://localhost:11434/v1`; override it with `--ollama-url` if Ollama is hosted elsewhere. Model-generated results may vary between runs.

## Rebuild the report

Open `Braess_Langroid_Report.ipynb` from the repository and run its cells. It reads the committed CSV summaries and writes the comparison figures and KPI table to `report_outputs/`.

## License

No license is granted by this repository. Ask the author before reusing or redistributing its code.
