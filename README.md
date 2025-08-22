# Axiomator: Building Real Machine Intelligence with Rules and Examples

Axiomator is a compact, deterministic, multimodal AI architecture that learns like a human: by ingesting rules and worked examples, not by guessing from patterns. It does not need GPUs, trillion-parameter models, or massive datasets. It is a machine that embodies knowledge directly, rather than a computer that tries to approximate it.

This repository is a single-file README that explains the concept, architecture, and provides executable-style examples you can adapt in any language such as Python. No other files are required.

***

## Why We’ve Been Building AI Wrong

Modern AI mostly optimizes guessing:
- It treats parameters as “knowledge,” when they are just weighted correlations.
- It requires huge compute to approximate methods that could be directly encoded as rules.
- It hallucinates because multiple statistical tendencies conflict, forcing resolution by probability rather than logic.
- It uses tool-calling as if intelligence were a collection of external apps instead of an integrated capability.

Real training is not “exposing a model to more data.” Real training is teaching methods with examples and rules, exactly how humans learn:
- Show the principle.
- Work through a few examples step by step.
- Apply deterministically.

When the rules and examples are embedded, the system stops guessing and simply executes. Determinism replaces probability. Reliability replaces hallucination. Efficiency replaces brute force.

***

## What Axiomator Is

Axiomator is:
- A single integrated intelligence engine that executes rules and examples deterministically.
- Modular and multimodal by design: text, images, audio, video, control, and reasoning share one rule engine.
- File-embeddable: rules and examples can be serialized to a single model file format such as safetensors for deployment.
- Extensible: size grows with knowledge, not with arbitrary parameter counts.
- Hardware-light: runs on commodity CPUs because it uses lookups, unification, and symbolic execution instead of vast tensor math.

Axiomator is not:
- A statistical pattern matcher that infers methods from millions of samples.
- A cascade of separate neural networks that must be fused.
- A tool caller that “uses a calculator.” It is the calculator, the reasoner, and the controller.

***

## Why This Should Be Implemented Now

- Deterministic correctness: In domains like math, logic, safety-critical control, and compliance, guessing is unacceptable.
- Efficiency: Rules and examples run on CPUs at low power with predictable latency.
- Maintainability: Knowledge is auditable. Update a rule file to fix behavior. No retraining cycles.
- Democratization: A full K‑12 tutor, a robotics controller, and a multimodal assistant can fit in tens of gigabytes and run on consumer hardware.
- Composability: Add domains by adding modules of rules and examples; no interference or forgetting.
- Safety and governance: Explicit rules support verification, provenance, and controlled capability growth.

***

## Core Principles

1. Rules over parameters  
   - Encode the method, not just the mapping.  
   - If a result depends on a law or theorem, embed that law or theorem.

2. Examples over datasets  
   - Curate minimal, canonical worked examples that demonstrate how rules are applied.  
   - Focus on coverage and clarity rather than scale.

3. Deterministic execution over probabilistic decoding  
   - Use a rule engine with unambiguous selection and conflict resolution based on precedence and applicability.  
   - No random sampling. No temperature. No approximate decoding.

4. Unified engine over tool soup  
   - Math, logic, perception, and control share the same engine semantics: match, bind, apply, reduce, and commit.

5. File-embedded knowledge  
   - Serialize rules and examples into a portable model file.  
   - Load once; execute everywhere.

***

## High-Level Architecture

- Core Engine (CE):  
  - Pattern matcher and unifier  
  - Rule selector with explicit precedence and guards  
  - Executor for transformations, derivations, and action plans  
  - Memory for bindings, subgoals, and proofs

- Knowledge Modules (KM):  
  - Math and Logic Axioms  
  - Language and Grammar Rules  
  - Vision Examples and Object Models  
  - Audio and Phonetic Patterns  
  - Procedural Control and Motion Primitives  
  - Cross-modal Alignment Rules

- Example Libraries (EL):  
  - Minimal worked examples per rule group  
  - Edge-case exemplars  
  - Strategy patterns: how to choose methods for problem classes

- Serializer  
  - Compiles KMs and ELs into a single model file (e.g., safetensors-style tensor blocks or any structured binary)  
  - Includes indices for fast lookup and precedence tables

- I/O Adapters  
  - Text, image, audio, video, and control inputs normalized to CE facts  
  - Outputs generated via deterministic composition

***

## Multimodal In One File

Suggested internal layout for a single model file (illustrative):

- Core_Engine(Main Neural Network): Any
- Text_Rules: 2 GB
- Image_Examples: 3 GB
- Audio_Patterns: 1 GB
- Video_Scripts: 2 GB
- Control_Procedures: 1 GB
- Cross_Modal_Rules: 500 MB
- And More

Total: ~7.6 GB to 20 GB for a robust assistant that handles language, perception, and planning without guessing. Sizes scale with actual knowledge, not arbitrary parameter counts.

***

## How It Works

1. Normalize input into facts  
   Example: “Differentiate x^3” → facts: intent(diff), expr(pow(x,3))

2. Match applicable rules  
   - Find rules whose guards and patterns match the facts.

3. Bind variables and apply transformations  
   - Unify pattern variables with input symbols; compute outputs.

4. If needed, decompose into subgoals  
   - Choose strategy patterns that break problems into solvable parts.

5. Produce final solution and proof trace  
   - Every step is explainable and reproducible.

6. Update working memory  
   - Cache intermediate results; optionally add new examples learned from executed steps.

***

## Examples

All examples below are expressed in a simple pseudo-DSL and Python-like pseudo-code to illustrate how Axiomator would work. You can implement similar structures in real Python.

### Example 1: Calculus Differentiation

Rule:
- If diff(x^n) with n as constant then n*x^(n-1)

Example entries:
- Input: diff(x^3) → 3*x^2
- Input: diff(5*x) → 5
- Input: diff(sin(x)) → cos(x)

Python-style demonstration:

```python
# Rule schema
rule_power_diff = {
    "match": {"op": "diff", "arg": {"op": "pow", "base": "x", "exp": {"const": True}}},
    "apply": lambda x, n: {"op": "mul", "args": [n, {"op": "pow", "base": "x", "exp": n-1}]},
    "guards": [lambda n: isinstance(n, (int, float))]
}

# Worked example application
def apply_diff(expr):
    if expr == {"op": "diff", "arg": {"op": "pow", "base": "x", "exp": 3}}:
        return {"op": "mul", "args": [3, {"op": "pow", "base": "x", "exp": 2}]}
    # Other rules: linearity, trig, product, chain, etc.
```

No probability, no sampling, no hallucination. Just rule match and apply.

### Example 2: Algebraic Simplification Strategy

Strategy pattern:
- Prefer constant folding
- Then factorization
- Then cancellation
- Maintain canonical ordering

Example:
- Input: (2*x + 3*x) → 5*x  
- Input: (x/x) with x ≠ 0 → 1

```python
def simplify(expr):
    expr = constant_fold(expr)
    expr = combine_like_terms(expr)
    expr = factor(expr)
    expr = cancel_valid(expr)
    return canonicalize(expr)
```

Precedence is explicit. Changes are explainable.

### Example 3: Vision Recognition by Examples

Object model:
- Chair has seat plane, backrest plane, support legs with approximate geometry and relative placement tolerances.

Example view embeddings:
- Store a few canonical 3D-2D projections with tolerances.

Recognition flow:
- Extract primitives from image (edges, planes, keypoints)  
- Match to embedded object templates with tolerances  
- Verify constraints: dimensions, relations, stability

```python
def recognize_chair(features):
    for template in CHAIR_TEMPLATES:
        if template.matches(features, tolerance=template.tolerance):
            return {"object": "chair", "confidence": 1.0}
    return {"object": None}
```

No guessing. If constraints match, it is a chair. If not, it is not.

### Example 4: Procedural Control by Embedded Examples

Walking primitive:
- Shift weight → lift leg A → place at target → stabilize → repeat with leg B  
- Balance rule enforces center-of-mass above support polygon.

```python
def walk_to(target):
    while not at_target(target):
        shift_weight(left=True)
        step(leg="right", to=next_footfall(target))
        stabilize()
        shift_weight(left=False)
        step(leg="left", to=next_footfall(target))
        stabilize()
```

This is a deterministic sequence governed by stability rules. Not trained. Not guessed.

### Example 5: Cross-Modal Instruction Following

Instruction: “Pick up the red cup on the table.”

- Parse intent → find objects matching “cup” with color=red → confirm on “table” surface → plan grasp using cup geometry primitives → execute

```python
plan = [
    "detect(objects={'type':'cup','color':'red','support':'table'})",
    "select(best_reachable(object))",
    "plan_grasp(object, approach='side', grip='cylindrical')",
    "execute_grasp()",
    "lift(height=0.2)"
]
```

Every step is a rule-driven decision.

***

## Learning New Examples Without Training

Axiomator “learns” by adding examples and rules, not by gradient descent.

- Add a new differentiation pattern with a worked example.
- Add a new object template with its geometric constraints.
- Add a new control primitive with safety guards.

In Python-like terms:

```python
knowledge_base.add_rule(rule_schema, worked_examples=[ex1, ex2])
knowledge_base.add_template("chair_v2", geometry=geom_spec, tolerance=spec)
knowledge_base.add_procedure("open_door", steps=door_steps, guards=safety_checks)
knowledge_base.serialize("Axiomator.safetensors")
```

This is education, not training. The system’s capability expands deterministically with every addition.

***

## File Embedding and Deployment

- Compile rules, examples, indices, and precedence tables into a single binary model file.
- The file can use a safetensors-like structure for portability and safety.
- At runtime:
  - Load Core Engine
  - Memory-map knowledge sections
  - Execute with zero stochasticity

Benefits:
- Drop-in deployment to existing model runners
- Versioned, auditable knowledge
- No GPUs required

***

## Roadmap You Can Implement Today

- Start with Math and Logic module: differentiation, algebra, proof tactics with 1000 high-quality worked examples.
- Add Language module: grammar, composition patterns, explanation templates.
- Add Vision module: canonical templates for common objects with geometric constraints.
- Add Control module: procedural primitives governed by safety and stability rules.
- Package all into a single model file.
- Expose a simple API:
  - solve(expression)
  - explain(problem)
  - recognize(image)
  - plan(task)
  - execute(plan)

***

## FAQ

- What about creativity  
  - Creativity emerges from recombining rules and examples across domains under explicit search strategies. The system can discover new patterns, then promote stable ones into new rules.

- What about ambiguity in language  
  - Use disambiguation rules and clarification strategies. Deterministic does not mean rigid; it means choices are explicit and auditable.

- Can it scale to new domains  
  - Yes. Add modules with rules and worked examples. No retraining. No catastrophic forgetting.

- Will it replace neural nets entirely  
  - In many safety-critical and method-driven domains, yes. Where perception requires robust low-level feature extraction, hybrid front-ends can normalize sensory input into facts before deterministic reasoning.

***

## Contributing

- Propose new rule schemas with clear guards and applicability notes.
- Contribute minimal, canonical worked examples that teach method, not just answers.
- Add cross-modal bindings that connect perception to action via explicit constraints.
- Document precedence and conflict resolution for every rule family.

***

## License

Apache 2.0
***

## Closing Thought

Intelligence does not have to be expensive, probabilistic, or fragile. It can be compact, certain, and teachable. Axiomator is about building machines that are what they do: the calculator that calculates, the reasoner that reasons, the perceiver that perceives, all by design rather than by chance.


