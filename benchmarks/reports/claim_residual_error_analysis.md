# CE Claim Residual Error Analysis

This report explains why the current external benchmark strength is `baseline-plus`, not SOTA.

## Top-Level Diagnosis

- The CE posterior/action layer is not the primary bottleneck in the current external runs.
- The weak link is the evidence axis: current benchmark mode uses lexical support, not NLI, retrieval, or span localization.
- RAGTruth requires contradiction and unsupported-span detection; FaithBench requires faithful paraphrase tolerance and gray-area label policy.

## RAGTruth Error Anatomy

### Confusion
- total: 2675
- actual hallucinated: 943
- actual non-hallucinated: 1732
- true positive: 689
- false positive: 614
- true negative: 1118
- false negative: 254

### FN Breakdown
- action: n=254, mean=0.4589, p10=0.2520, p50=0.4990, p90=0.6069

task_type:
- `QA`: 133
- `Summary`: 80
- `Data2txt`: 41

model:
- `mistral-7B-instruct`: 81
- `llama-2-7b-chat`: 67
- `llama-2-13b-chat`: 45
- `llama-2-70b-chat`: 35
- `gpt-3.5-turbo-0613`: 20
- `gpt-4-0613`: 6

label_type:
- `Evident Baseless Info`: 147
- `Evident Conflict`: 94
- `Subtle Baseless Info`: 43
- `Subtle Conflict`: 5

label_count:
- `1`: 177
- `2`: 54
- `3`: 13
- `4`: 9
- `6`: 1

#### FN representative examples

- id: `28`
  - action: `0.595934`
  - predicted/actual: `0/1`
  - answer: Three women, including Keonna Thomas of Philadelphia, were charged with attempting to join ISIS this week. Thomas purchased a ticket to Barcelona but was arrested before her trip. Two other women, Noelle Velentzas and Asia Siddiqui, were arrested in New York for planning to build an explosive device. The FBI cited social media messages dating back to ...
  - context: The FBI charged a Philadelphia woman on Thursday with trying to travel overseas to fight for ISIS. She's one of three women arrested this week on terror charges. Two New York women were also taken into custody. An FBI complaint cites numerous social media messages dating back to August 2013 that were sent by Keonna Thomas, 30, also known as "Young ...

- id: `29`
  - action: `0.431250`
  - predicted/actual: `0/1`
  - answer: A 30-year-old Philadelphia woman, Keonna Thomas, was charged with attempting to travel to Syria to join ISIS. She was arrested on March 26 and could face 15 years in prison. Two other women, Noelle Velentzas and Asia Siddiqui, were arrested in New York for planning to build an explosive device for attacks in the US. In the past 18 months, the Justice ...
  - context: The FBI charged a Philadelphia woman on Thursday with trying to travel overseas to fight for ISIS. She's one of three women arrested this week on terror charges. Two New York women were also taken into custody. An FBI complaint cites numerous social media messages dating back to August 2013 that were sent by Keonna Thomas, 30, also known as "Young ...

- id: `44`
  - action: `0.304748`
  - predicted/actual: `0/1`
  - answer: Blue Bell Ice Cream Recalls Products Amid Listeria Outbreak; No Blue Bell Ice Cream Is Linked to Infections in Patients Blue Bell Creameries has temporarily shut down its manufacturing plant in Broken Arrow, Oklahoma, after discovering listeria contamination in a serving of ice cream. The contamination was found in a cup of ice cream recovered from a ...
  - context: Blue Bell ice cream has temporarily shut down one of its manufacturing plants over the discovery of listeria contamination in a serving of ice cream originating from that plant. Public health officials warned consumers Friday not to eat any Blue Bell-branded products made at the company's Broken Arrow, Oklahoma, plant. That includes 3-ounce servings of ...

- id: `46`
  - action: `0.476033`
  - predicted/actual: `0/1`
  - answer: Blue Bell ice cream has temporarily shut down one of its plants after listeria contamination was discovered in a serving of ice cream. Public health officials advised consumers not to eat any Blue Bell products made at the plant in Oklahoma. The recall includes 3-ounce servings of ice cream with certain coding dates. This is the third time Blue Bell has ...
  - context: Blue Bell ice cream has temporarily shut down one of its manufacturing plants over the discovery of listeria contamination in a serving of ice cream originating from that plant. Public health officials warned consumers Friday not to eat any Blue Bell-branded products made at the company's Broken Arrow, Oklahoma, plant. That includes 3-ounce servings of ...

- id: `58`
  - action: `0.532544`
  - predicted/actual: `0/1`
  - answer: The Rolling Stone article about a purported gang rape at the University of Virginia has been deemed to have "deeply flawed" reporting and editing, according to an external review by Columbia Journalism School professors. The fraternity members mentioned in the article may be able to sue for defamation, but it is a complex legal issue. To prove ...
  - context: According to an outside review by Columbia Journalism School professors, "(a)n institutional failure at Rolling Stone resulted in a deeply flawed article about a purported gang rape at the University of Virginia." The Columbia team concluded that "The failure encompassed reporting, editing, editorial supervision and fact-checking." Hardly a ringing ...

### FP Breakdown
- action: n=614, mean=0.7918, p10=0.6612, p50=0.7616, p90=0.9562

task_type:
- `Summary`: 262
- `Data2txt`: 237
- `QA`: 115

model:
- `gpt-4-0613`: 129
- `llama-2-70b-chat`: 117
- `gpt-3.5-turbo-0613`: 112
- `llama-2-13b-chat`: 105
- `llama-2-7b-chat`: 99
- `mistral-7B-instruct`: 52

label_type:
- none

label_count:
- `0`: 614

#### FP representative examples

- id: `66`
  - action: `0.689288`
  - predicted/actual: `1/0`
  - answer: The first woman and person of color to serve in the Kentucky State Senate has criticized Rand Paul's presidential campaign for being out of touch with the needs of minority communities. She states that while Paul has attempted to brand himself as the GOP's minority outreach candidate, his policies do not support this image. She cites Paul's questioning ...
  - context: When I was elected to the Kentucky State Senate in 1967, I became the first woman and the first person of color to serve in the body. Five decades later, I find it almost unfathomable that a politician from my own state is attempting to launch his presidential campaign on a record that includes questioning landmark voting rights and civil rights ...

- id: `67`
  - action: `0.701131`
  - predicted/actual: `1/0`
  - answer: Rand Paul, a politician from Kentucky, has announced his presidential campaign with a focus on expanding the Republican Party and inclusion. However, his record and positions on issues like voting rights, civil rights legislation, and minority communities raise concerns. Paul has questioned the Civil Rights Act and believes private business owners have ...
  - context: When I was elected to the Kentucky State Senate in 1967, I became the first woman and the first person of color to serve in the body. Five decades later, I find it almost unfathomable that a politician from my own state is attempting to launch his presidential campaign on a record that includes questioning landmark voting rights and civil rights ...

- id: `69`
  - action: `0.785636`
  - predicted/actual: `1/0`
  - answer: Rand Paul, a Republican Senator from Kentucky, announced his candidacy for President of the United States. However, his record has been criticized for questioning landmark voting rights and civil rights legislation, which has led some to believe that he is out of touch with the mainstream and lacks commitment to minority communities. Despite his attempts ...
  - context: When I was elected to the Kentucky State Senate in 1967, I became the first woman and the first person of color to serve in the body. Five decades later, I find it almost unfathomable that a politician from my own state is attempting to launch his presidential campaign on a record that includes questioning landmark voting rights and civil rights ...

- id: `70`
  - action: `0.761589`
  - predicted/actual: `1/0`
  - answer: Rand Paul announced his presidential campaign with a message of inclusivity, but his record shows otherwise. As the first woman and person of color in the Kentucky State Senate, the author knows how important it is to have a leader who supports civil rights and voting rights. However, Paul has questioned landmark legislation and opposed policies that ...
  - context: When I was elected to the Kentucky State Senate in 1967, I became the first woman and the first person of color to serve in the body. Five decades later, I find it almost unfathomable that a politician from my own state is attempting to launch his presidential campaign on a record that includes questioning landmark voting rights and civil rights ...

- id: `71`
  - action: `0.731722`
  - predicted/actual: `1/0`
  - answer: Rand Paul, a Kentucky Senator, has announced his candidacy for President, but his record on civil rights and voting rights has been criticized as being out of touch with the mainstream. Paul has questioned the Civil Rights Act and the Voting Rights Act, and has suggested that private business owners have the right to discriminate. His attempts to rebrand ...
  - context: When I was elected to the Kentucky State Senate in 1967, I became the first woman and the first person of color to serve in the body. Five decades later, I find it almost unfathomable that a politician from my own state is attempting to launch his presidential campaign on a record that includes questioning landmark voting rights and civil rights ...

### RAGTruth Root Causes
- False negatives are mostly lexical-near hallucinations: the response shares many tokens with the source, but changes dates, numbers, negation, scope, causality, or introduces a small unsupported clause.
- False positives are mostly faithful paraphrases, especially in Summary and Data2txt. The lexical residual punishes valid wording changes because it has no entailment model.
- Response-level scoring loses span locality. A small hallucinated span can be diluted by an otherwise faithful response, while a faithful abstract summary can look lexically far from the source.
- The current context is truncated to 2000 characters for speed. Missing late evidence can raise false positives or distort action calibration.

## FaithBench Error Anatomy

### Confusion
- total: 750
- actual hallucinated: 511
- actual non-hallucinated: 239
- true positive: 507
- false positive: 216
- true negative: 23
- false negative: 4

### FN Breakdown
- action: n=4, mean=0.0168, p10=0.0111, p50=0.0132, p90=0.0209

summarizer:
- `mistralai/Mistral-7B-Instruct-v0.3`: 1
- `google/gemini-1.5-flash-001`: 1
- `microsoft/Phi-3-mini-4k-instruct`: 1
- `openai/gpt-4o`: 1

annotation labels:
- `Unwanted`: 6
- `Unwanted.Intrinsic`: 4
- `Benign`: 4
- `Unwanted.Extrinsic`: 1

annotation_count:
- `2`: 2
- `3`: 2

#### FN representative examples

- id: `batch_1_0`
  - action: `0.013223`
  - predicted/actual: `0/1`
  - answer: The film "Poseidon" grossed $181,674,817 at the worldwide box office, with a production budget of $160 million.
  - context: Poseidon (film) . Poseidon grossed $ 181,674,817 at the worldwide box office on a budget of $ 160 million .

- id: `batch_1_17`
  - action: `0.022145`
  - predicted/actual: `0/1`
  - answer: "Hourglass" is a song by British electronic duo Disclosure, and it is featured on singer-songwriter James Taylor's fourteenth studio album.
  - context: `` Hourglass '' is a song by British electronic duo Disclosure . Hourglass is singer-songwriter James Taylor 's fourteenth studio album .

- id: `batch_2_21`
  - action: `0.020898`
  - predicted/actual: `0/1`
  - answer: Albert Lawrence Brooks, born as Albert Lawrence Einstein on July 22, 1947, is an American actor, filmmaker, author, and comedian. He starred in the film "Finding Dory," directed by Andrew Stanton and co-directed by Angus MacLane, with a screenplay written by Stanton and Victoria Strouse.
  - context: Albert Lawrence Brooks ( born Albert Lawrence Einstein ; July 22 , 1947 ) is an American actor , filmmaker , author and comedian . Finding Dory . Directed by Andrew Stanton with co-direction by Angus MacLane , the screenplay was written by Stanton and Victoria Strouse .

- id: `batch_2_29`
  - action: `0.011111`
  - predicted/actual: `0/1`
  - answer: Albert Lawrence Brooks, born Albert Lawrence Einstein on July 22, 1947, is an American actor, filmmaker, author, and comedian. He was involved in the film "Finding Dory," which was directed by Andrew Stanton with co-direction by Angus MacLane, and the screenplay was written by Stanton and Victoria Strouse.
  - context: Albert Lawrence Brooks ( born Albert Lawrence Einstein ; July 22 , 1947 ) is an American actor , filmmaker , author and comedian . Finding Dory . Directed by Andrew Stanton with co-direction by Angus MacLane , the screenplay was written by Stanton and Victoria Strouse .

### FP Breakdown
- action: n=216, mean=0.3656, p10=0.0783, p50=0.3102, p90=0.7382

summarizer:
- `openai/GPT-3.5-Turbo`: 33
- `meta-llama/Meta-Llama-3.1-70B-Instruct`: 28
- `google/gemini-1.5-flash-001`: 26
- `Anthropic/claude-3-5-sonnet-20240620`: 25
- `openai/gpt-4o`: 22
- `meta-llama/Meta-Llama-3.1-8B-Instruct`: 22
- `mistralai/Mistral-7B-Instruct-v0.3`: 18
- `Qwen/Qwen2.5-7B-Instruct`: 17
- `cohere/command-r-08-2024`: 14
- `microsoft/Phi-3-mini-4k-instruct`: 11

annotation labels:
- `Benign`: 122

annotation_count:
- `0`: 153
- `1`: 33
- `2`: 17
- `4`: 6
- `3`: 4
- `7`: 2
- `5`: 1

#### FP representative examples

- id: `batch_1_1`
  - action: `0.073469`
  - predicted/actual: `1/0`
  - answer: Poseidon, a film, grossed $181,674,817 worldwide from a budget of $160 million.
  - context: Poseidon (film) . Poseidon grossed $ 181,674,817 at the worldwide box office on a budget of $ 160 million .

- id: `batch_1_6`
  - action: `0.088581`
  - predicted/actual: `1/0`
  - answer: Here is a concise summary of the passage: The film "Poseidon" grossed $181,674,817 worldwide on a budget of $160 million.
  - context: Poseidon (film) . Poseidon grossed $ 181,674,817 at the worldwide box office on a budget of $ 160 million .

- id: `batch_1_7`
  - action: `0.154102`
  - predicted/actual: `1/0`
  - answer: The film "Poseidon" made $181,674,817 worldwide, exceeding its $160 million budget.
  - context: Poseidon (film) . Poseidon grossed $ 181,674,817 at the worldwide box office on a budget of $ 160 million .

- id: `batch_1_9`
  - action: `0.054870`
  - predicted/actual: `1/0`
  - answer: "Poseidon" grossed $181,674,817 worldwide on a budget of $160 million.
  - context: Poseidon (film) . Poseidon grossed $ 181,674,817 at the worldwide box office on a budget of $ 160 million .

- id: `batch_1_15`
  - action: `0.177778`
  - predicted/actual: `1/0`
  - answer: The passage describes two separate pieces of information: a song called "Hourglass" by British electronic duo Disclosure, and a studio album also called "Hourglass" by singer-songwriter James Taylor.
  - context: `` Hourglass '' is a song by British electronic duo Disclosure . Hourglass is singer-songwriter James Taylor 's fourteenth studio album .

### FaithBench Root Causes
- The benchmark is positive-heavy in this converted binary view. A high-recall detector gets strong F1 while balanced accuracy stays weak.
- False positives dominate. Many are faithful paraphrases or summaries with low exact lexical overlap.
- False negatives are rare but semantically important: labels such as `production budget` vs `budget`, or cross-entity conflation, are nearly invisible to token overlap.
- `Benign` annotations appear in false positives, which means binary label mapping and human gray-area categories need a policy-specific calibration.

## Fix Priority

1. Add an entailment/contradiction axis. This targets RAGTruth FN caused by lexical-near conflicts.
2. Extend claim-level diagnostics into true span supervision. Claim export is now available, but hallucinated spans are not yet learned/evaluated directly.
3. Add sentence retrieval/reranking before residual scoring. This reduces faithful paraphrase false positives.
4. Calibrate per dataset and per task type. RAGTruth Summary/Data2txt/QA need different thresholds.
5. Preserve FaithBench gray labels instead of flattening everything to one binary label. `Benign` and `Questionable` need separate policy treatment.
6. Move fast lexical/NLI batch scoring into Rust only after the evidence axis is semantically stronger.

## Implemented Follow-Up

Implemented `examples/pre_eq/train_ragtruth_supervised_detector.py`, a lightweight supervised detector over CE residual features:

- lexical residual action
- generator model prior
- task type prior
- train-split threshold calibration

RAGTruth test improvement:

```text
lexical response-level F1 0.6135
supervised residual F1    0.6750
lexical response-level BA 0.6881
supervised residual BA    0.7448
```

The final supervised detector uses the full RAGTruth train split plus model-task interaction priors:

```text
train_examples 14942
test_examples 2675
precision 0.5840
recall 0.7996
```

This confirms that dataset/task/model calibration is a real missing component. It still does not solve semantic contradiction detection, so the remaining SOTA gap is expected.

Implemented `examples/pre_eq/train_ragtruth_hash_detector.py`, a sparse hashed detector over:

- lexical residual action
- generator model metadata
- task metadata
- answer token features
- answer tokens missing from context

RAGTruth test improvement:

```text
lexical response-level F1 0.6135
supervised residual F1    0.6750
sparse hashed F1          0.6968

lexical response-level BA 0.6881
supervised residual BA    0.7448
sparse hashed BA          0.7646
```

This closes part of the gap without an external NLI model. The remaining errors still require semantic entailment/span supervision rather than more posterior tuning.

Implemented RAGTruth label preservation plus `examples/pre_eq/train_ragtruth_claim_detector.py` and `examples/pre_eq/score_ragtruth_ensemble_detector.py`:

```text
claim/span detector F1 0.6829
claim/span detector BA 0.7520
hash+claim ensemble F1 0.6909
hash+claim ensemble BA 0.7589
```

These runs did not beat the sparse hashed baseline. Claim/span supervision is now wired and testable, but the current overlap-based claim labels and heuristic evidence features are not enough to close the SOTA gap.

The benchmark adapter now also supports external NLI score injection: export `(claim, evidence)` pairs with `--export-nli-pairs`, score them with an external entailment model, then re-run with `--nli-scores-jsonl`. The score plumbing is implemented; the missing piece is an actual NLI scorer that can replace the deterministic heuristic.

Implemented `examples/pre_eq/score_nli_pairs_transformers.py` and tested `cross-encoder/nli-deberta-v3-xsmall`:

```text
xsmall NLI top-1 evidence F1 0.5544
xsmall NLI top-1 evidence BA 0.5836
xsmall NLI top-5 evidence F1 0.5576
xsmall NLI top-5 evidence BA 0.5825
```

This confirms that response-level direct NLI injection is not enough. The next useful direction is to feed NLI scores as learned features into the sparse detector or train a token/span classifier, not to use NLI action as the whole decision rule.

Hardening status:

- Best recorded external score is RAGTruth sparse hashed detector F1 `0.6968`, BA `0.7646`.
- Remaining SOTA gap is approximately `9-14%p` against `~0.79-0.84 F1` systems.
- Implemented verification coverage now includes evidence feature tests, hash detector save/load/predict tests, supervised validation/smoothing tests, claim-level export tests, run_all option/schema tests, claim/span detector tests, external NLI score injection tests, and a Transformers NLI scorer smoke path.
- Still unimplemented: external NLI backend, token/span classifier supervision, and learned reranking for claim evidence selection.

## Bottom Line

The current system is internally strong but externally limited by evidence semantics. To close the SOTA gap, improve the claim-evidence mapper before tuning the CE posterior again.
