# Project proposal

## 1. The problem and goal

**The problem and goal in my project:**

Many medicines act on a biological target that is present both in the body at
large and in the brain. When the molecule enters the brain, it produces central
side effects that have nothing to do with the therapeutic purpose. The classic
example is antihistamines: first-generation compounds such as diphenhydramine
block the histamine H1 receptor effectively, but they also cross the
blood-brain barrier and cause drowsiness. Second-generation antihistamines were
developed specifically to keep the same activity while staying out of the
brain. Designing a molecule that is deliberately restricted to the periphery is
therefore a real and recurring task in medicinal chemistry, and it is difficult
because the properties that control brain entry (lipophilicity, polar surface
area, charge, protein binding) are the same properties that control solubility,
metabolic stability and potency. Improving one usually damages another.

My project builds an agent that performs this task as a lead-optimisation loop.
Starting from a known sedating parent compound, the agent proposes structural
analogues, a computational pipeline predicts where each analogue distributes in
the body, and the agent uses that prediction to decide what to modify next.

**What I want to improve.** The predicted exposure of the molecule in the brain
relative to its exposure in blood plasma, while the rest of its drug-like
properties stay acceptable.

**How I will measure improvement.** The primary metric is the predicted
brain-to-plasma ratio of the area under the concentration-time curve,
$\mathrm{AUC}_{\text{brain}} / \mathrm{AUC}_{\text{plasma}}$, computed by a
physiologically based pharmacokinetic model. The target is a value below 0.1,
compared with the considerably higher value the same pipeline predicts for the
parent compound. Secondary metrics are the predicted fraction of total exposure
in the liver, kidney and fat compartments.

**Limits and requirements that must be respected.**

- The proposed structure must be a chemically valid molecule that parses in
  RDKit.
- Molecular weight at most 500 g/mol.
- At most three structural modifications away from the parent compound, so the
  result remains a recognisable analogue rather than an unrelated molecule.
- Synthetic accessibility score at most 4, so the molecule is plausibly
  makeable.
- No PAINS structural alerts (substructures known to cause false positives and
  non-specific activity).
- Predicted cardiac toxicity risk (hERG channel inhibition) below a fixed
  threshold.
- Predicted aqueous solubility above a fixed threshold.
- A budget of at most 50 pipeline evaluations per run, so the loop stays
  bounded.

**Validation anchor.** Before the agent is trusted, the pipeline must reproduce
a known ordering: it must predict higher brain exposure for diphenhydramine
than for cetirizine and loratadine. If it cannot reproduce a result that is
established pharmacology, its predictions for new molecules are not worth
reading.

**A short example for someone outside the field.** A sleeping pill and an
allergy pill can be built on very similar chemical skeletons. The difference is
often not what the molecule does, but where in the body it manages to go. The
goal here is to take a molecule that works but reaches the brain, and redesign
it so that it still works and no longer reaches the brain.

## 2. The simulator

**The simulator in my project:**

The ground is a pipeline of three layers. It takes one molecule, written as a
SMILES string (a standard one-line text notation for a chemical structure; for
example ethanol is `CCO`), and returns a quantitative prediction of where that
molecule distributes in the body.

**Layer 1 — molecular descriptors (RDKit).** RDKit is an open-source
cheminformatics library for Python. Given the SMILES string, it computes
structural and physicochemical descriptors by deterministic rules: molecular
weight, the octanol-water partition coefficient $\log P$, topological polar
surface area, hydrogen-bond donors and acceptors, rotatable bonds, structural
alerts and a synthetic accessibility score. This layer is exact and takes
milliseconds.

**Layer 2 — ADMET prediction (ADMET-AI).** ADMET-AI is an open-source,
pip-installable Python package built on a graph neural network trained on 41
public datasets from the Therapeutics Data Commons. It predicts absorption,
distribution, metabolism, excretion and toxicity endpoints. I need three of
them: the fraction of drug not bound to plasma proteins, total clearance, and
the hERG toxicity flag. The model weights are fixed at inference, so the same
input returns the same output. The package version will be pinned, because the
authors state that predictions differ between major versions.

**Layer 3 — physiologically based pharmacokinetic model (my own, in SciPy).**
This layer converts molecular properties into organ-level concentrations. The
body is represented as a set of well-mixed compartments (brain, liver, kidney,
gut, muscle, fat, rest of body, and blood), each with a physiological volume
$V_i$ and blood flow $Q_i$ taken from published tables. The tissue-to-plasma
partition coefficient $K_{p,i}$ of each compartment is estimated from
$\log P$, $pK_a$, the unbound fraction and the lipid, water and protein
composition of that tissue, using published correlation equations
(Poulin-Theil and Rodgers-Rowland). The resulting system of ordinary
differential equations, one mass balance per compartment,

$$V_i \frac{dC_i}{dt} = Q_i \left( C_{\text{art}} - \frac{C_i}{K_{p,i}} \right) - \mathrm{CL}_i \, C_i$$

is integrated with `scipy.integrate.solve_ivp`. Structurally this is a network
of continuous stirred-tank reactors with elimination in the liver and kidney,
which is material I already know from reactor design.

**Inputs.** A SMILES string, plus a configuration file in JSON or YAML holding
the dose, the dosing route, the simulated time window, the physiological
parameter set, and the constraint thresholds listed in section 1.

**Outputs.** A JSON object containing the computed descriptors, the predicted
ADMET endpoints, the concentration-time profile of each compartment, the area
under the curve for each organ, the brain-to-plasma ratio, and a pass or fail
result for each constraint with the reason for any failure.

**Does it run on my computer, with inputs from a text file?** Yes. All three
layers are free, open source and run locally on Ubuntu under WSL, with
configuration supplied as JSON. One run takes on the order of a second, so a
loop of dozens of evaluations fits comfortably in a session.

**What I still need to check.** First, that ADMET-AI installs cleanly on my
machine, since it depends on PyTorch and Chemprop and the download is large. If
it does not, the fallback is to estimate the unbound fraction from $\log P$
using a published correlation, which keeps the pipeline complete at the cost of
some accuracy. Second, which quantities should come from the machine-learning
layer and which should be derived mechanistically: ADMET-AI can also predict
volume of distribution directly, and taking that value while the
pharmacokinetic model independently predicts distribution would double-count
the same physics. My current intention is to take only the unbound fraction and
clearance from layer 2, and to derive all partition coefficients
mechanistically in layer 3.

**How the ground meets the six criteria.** It is importable in Python with a
callable interface and no graphical interface. Every component is free and open
source. It is deterministic, which I will assert in a test by running the same
molecule twice and comparing outputs exactly. A run is bounded and takes about
a second. It is testable with pytest: mass must be conserved across
compartments, a single compartment without elimination must reproduce the
analytical result $C = \text{dose}/V$, increasing the unbound fraction must
increase tissue exposure, and the known antihistamine ordering must be
reproduced. It is non-trivial, because no single call returns the answer: the
molecule must be constructed, evaluated across several conflicting criteria,
and revised.

## 3. The agent's role

**The agent in my project:**

The agent performs lead optimisation. It receives a design brief in words,
proposes a candidate analogue, reads the pipeline's full output, works out
which requirement is currently binding, and proposes the next analogue with a
stated chemical reason.

**What the agent may change.** The structure of the molecule, expressed as
edits to the SMILES string of the parent compound: adding, removing or
replacing substituents and functional groups.

**Allowed ranges.**

- Heavy atoms: 15 to 40.
- Molecular weight: up to 500 g/mol.
- Element set: carbon, nitrogen, oxygen, sulfur, fluorine, chlorine.
- At most three modifications from the parent compound.
- At most 50 pipeline evaluations per run.

The agent does not compute any property itself and does not decide whether a
molecule is valid. Validity, structural alerts and synthetic accessibility are
checked and enforced by the ground, which rejects a non-compliant proposal and
returns the reason. The agent reasons; the pipeline computes.

**How it uses the results.** Each evaluation returns organ-level exposures
together with the constraint report. The agent reads which constraint failed
and by how much, and selects a modification intended to move that quantity in
the required direction without breaking the constraints that currently pass.
For example, if brain exposure is still too high while solubility has room to
spare, a reasonable move is to add a polar group that raises polar surface area
and lowers lipophilicity. If the same change has already been tried and pushed
clearance out of range, the agent must try a structurally different approach.

**Could a simple program that tries options achieve the same goal?** Partly,
and I want to be accurate about this rather than overstate the case. A genetic
algorithm that mutates molecular graphs is a strong and well-established
baseline for molecular optimisation, and for a single fixed numerical objective
it might well match an agent. The language model has to earn its place
elsewhere, and I claim three specific contributions. First, it converts a brief
written in ordinary language into an explicit set of numerical constraints and
thresholds, which a search procedure cannot do because it must be given the
objective already formalised. Second, it supplies a chemical rationale for each
modification, so a chemist can audit the reasoning and not only the result,
whereas a genetic algorithm returns a structure with no explanation. Third,
when the brief changes partway through, for instance when a new toxicity
requirement is added, it can adapt the strategy without restarting the search.

To test this claim rather than assert it, the project will include a
random-mutation baseline as a negative control, run under the same evaluation
budget. If the agent does not beat it, that is a finding worth reporting.

**One question I still need to answer.** Several second-generation
antihistamines stay out of the brain largely because they are actively pumped
back out by the P-glycoprotein efflux transporter at the blood-brain barrier,
rather than because of passive permeability alone. A perfusion-limited
pharmacokinetic model with partition coefficients derived from lipophilicity
does not represent active efflux at all. I need to determine whether to add an
efflux term to the brain compartment, parameterised by a predicted
P-glycoprotein substrate flag, or to state explicitly that the model covers
passive distribution only and to restrict the claims accordingly. This decision
directly affects where the line falls between the model's reasoning and the
ground's computation.

## Explain your project to someone outside your field

**My project in simple words:**

When you take a medicine, it does not stay where you swallowed it. It enters
the bloodstream and spreads through the whole body, and it reaches different
organs in very different amounts depending on the shape and chemistry of the
molecule. Sometimes it reaches an organ where it is not wanted. Older allergy
tablets, for instance, make people sleepy because the molecule enters the brain
in addition to doing its intended job elsewhere. Chemists solve this by
redesigning the molecule so that it keeps working but no longer gets into the
brain. The difficulty is that every change has side effects of its own: a
change that keeps the molecule out of the brain may also make it impossible to
dissolve, or cause it to accumulate in body fat, or make it harmful to the
heart. There are far too many possible molecules to test in a laboratory, and
each laboratory test is slow and expensive.

I am building a computer pipeline that takes the description of a molecule and
estimates how much of it would end up in each organ. It does this in three
stages: it first calculates basic properties of the molecule from its
structure, then estimates how strongly the molecule sticks to proteins in blood
and how quickly the body destroys it, and finally represents the body as a set
of connected tanks, one for each organ, with blood flowing between them, and
solves the equations describing how the substance moves between those tanks
over time. On top of this pipeline I am putting an artificial-intelligence
assistant of the kind that can read and write text and run programs. The
assistant is told the design goal in plain words. It suggests a molecule, the
pipeline calculates where that molecule would go, the assistant reads the
result, works out which requirement is currently being violated, and suggests a
better molecule. It repeats this until it finds candidates that satisfy every
requirement. Crucially, the assistant never calculates anything itself, because
this type of program is unreliable with numbers; all the arithmetic is done by
the tested pipeline. The assistant contributes the chemical judgement about
what to try next, and an explanation of why, which a human chemist can then
check.
