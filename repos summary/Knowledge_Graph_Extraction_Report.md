# Knowledge Graph Extraction from Text — A Comparative Benchmark of Triplet Extractors

**Project Report — STAT 451 Final Project**

**Authors of original work:** Yuxin He, Yiwu Zhong, Ching-Wen Wang
**Course:** STAT 451 (Introduction to Machine Learning), University of Wisconsin–Madison
**Tools compared:** spaCy (custom rule-based), Stanford OpenIE, Stanford Scene Graph Parser (via SPICE)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals and Motivation](#2-project-goals-and-motivation)
3. [Background — Knowledge Graphs and Triplets](#3-background--knowledge-graphs-and-triplets)
4. [Dataset — Custom-Collected Sentences from Three Domains](#4-dataset--custom-collected-sentences-from-three-domains)
5. [Tool 1 — spaCy with a Rule-Based Dependency Parser](#5-tool-1--spacy-with-a-rule-based-dependency-parser)
6. [Tool 2 — Stanford OpenIE](#6-tool-2--stanford-openie)
7. [Tool 3 — Stanford Scene Graph Parser via SPICE](#7-tool-3--stanford-scene-graph-parser-via-spice)
8. [Evaluation Methodology — Precision, Recall, F1 with Lemmatized Matching](#8-evaluation-methodology--precision-recall-f1-with-lemmatized-matching)
9. [Results](#9-results)
10. [Comparative Analysis and Discussion](#10-comparative-analysis-and-discussion)
11. [Visualization — Triplets to Knowledge Graphs](#11-visualization--triplets-to-knowledge-graphs)
12. [Conclusions and Future Work](#12-conclusions-and-future-work)
13. [Appendix — File Inventory](#13-appendix--file-inventory)

---

## 1. Executive Summary

This project addresses the problem of **automatic knowledge graph extraction from natural language text** — specifically, the upstream subproblem of **triplet extraction**, in which a sentence is decomposed into one or more `<subject, predicate, object>` tuples. The project's central contribution is a controlled empirical comparison of three widely-used triplet extractors — **spaCy with a custom dependency-parser-based rule system**, **Stanford OpenIE**, and the **Stanford Scene Graph Parser** (accessed via SPICE) — on a self-curated dataset spanning three textual domains: news articles, novels, and image captions.

The team manually annotated **~300 sentences** (100 per domain) with ground-truth triplets, then ran each tool on each domain and computed precision, recall, and F1 score under two matching schemes: (a) exact lemma matching and (b) loose lemma matching with substring overlap.

**Headline F1 results (loose matching, the more lenient and arguably more practically meaningful metric):**

| Method | News | Novel | Caption |
|---|---|---|---|
| spaCy (custom rules) | 38.25 | 38.25 | 53.55 |
| **Stanford OpenIE** | **55.82** | **55.82** | **61.77** |
| SPICE (Scene Graph Parser) | 29.43 | 24.23 | 49.85 |

**Three substantive findings emerge:**

1. **Stanford OpenIE is the best general-purpose extractor.** It dominates on news and novel domains and ties for the best on image captions.
2. **All three tools perform best on image captions.** Captions are syntactically simple, with explicit `<entity, action, entity>` structure that aligns closely with the triplet representation.
3. **The Scene Graph Parser is highly domain-specific.** Trained on visual-grounding caption data, it transfers poorly to news and novel text — exhibiting a 12-percentage-point F1 gap between captions and news, and 25 percentage points between captions and novels.

The project is a textbook case of dataset-curation and tool-benchmarking work — narrower in scope than supervised model-building, but methodologically rigorous in its evaluation protocol and substantively useful in producing an honest, domain-stratified comparison of off-the-shelf NLP tooling.

---

## 2. Project Goals and Motivation

### 2.1 The Information Extraction Problem

Knowledge graphs — structured representations of entities and relationships — are foundational to modern information systems: web search (Google's Knowledge Graph), question answering (Wolfram Alpha, modern LLM grounding), data integration (enterprise knowledge bases), and visual reasoning (scene graphs in computer vision). Constructing knowledge graphs at scale requires automated information extraction from unstructured text, and the central building block of this process is **triplet extraction**:

$$
\text{sentence} \;\longrightarrow\; \{\langle \text{subject}_i, \text{predicate}_i, \text{object}_i \rangle\}_{i=1}^{n}
$$

For example, the sentence *"Barack Obama was born in Hawaii"* should produce the triplet `<Barack Obama, was born in, Hawaii>`.

### 2.2 The Specific Question Posed

The team's framing is honest and practically motivated:

> *Since previous models were trained and evaluated on different datasets and settings, it's not sure which model is the suitable one in practice.*

This is the **out-of-distribution generalization** question for NLP tooling, applied to triplet extraction specifically. Each existing tool has been benchmarked by its authors on a domain that suits it (OpenIE on news, the Scene Graph Parser on captions). What happens when these tools are run on data they were not optimized for?

### 2.3 Project Objectives

1. **Curate a multi-domain evaluation dataset** with ground-truth triplet annotations, since none of the three tools were originally evaluated on a unified benchmark.
2. **Implement or wrap each tool** in a uniform interface that produces triplets in a common output format.
3. **Define a rigorous evaluation protocol** that handles morphological variation (lemmatization) and partial matches (substring overlap), allowing fair comparison.
4. **Run each tool on each domain** and compute precision, recall, F1.
5. **Identify the best-performing tool per domain** and provide actionable recommendations.
6. **Visualize the resulting knowledge graph** for the best tool, demonstrating end-to-end pipeline functionality.

### 2.4 Theoretical Expectations

Going in, the team's expectation is that performance will vary substantially across domains because:

- **OpenIE** is a general-purpose tool trained primarily on news/Wikipedia; expected to perform well on news, less well on captions.
- **The Scene Graph Parser** was trained on image-caption data; expected to dominate on captions but transfer poorly elsewhere.
- **spaCy + custom rules** is the simplest baseline and depends entirely on the quality of the hand-crafted rule set; expected to be competitive but not best.

The actual results (Section 9) confirm parts of this picture but contain surprises — most notably OpenIE's strength on captions and the Scene Graph Parser's weakness on novels.

---

## 3. Background — Knowledge Graphs and Triplets

### 3.1 Formal definition

A **knowledge graph** $G = (V, E, R)$ consists of:
- A set of **nodes** $V$ representing entities (people, places, things, abstract concepts).
- A set of **labeled directed edges** $E \subseteq V \times R \times V$, where each edge carries a relation label $r \in R$.

Equivalently — and more useful for extraction work — a knowledge graph is a set of triplets $\{\langle s, p, o \rangle\}$ where $s, o \in V$ are the subject and object entities, and $p \in R$ is the predicate (the relation between them).

### 3.2 Why triplets?

The triplet representation has several appealing properties:

1. **Decomposability.** Complex sentences split cleanly into multiple triplets (e.g., *"London is the capital of England and is on the Thames"* → `<London, is, capital of England>`, `<London, on, Thames>`).
2. **Composability.** Triplets from many sentences merge into a single knowledge graph by node identification (any mention of "London" in either triplet collapses to a single node).
3. **Computational tractability.** Triplets are the unit of representation for SPARQL queries on RDF stores, for embedding-based methods (TransE, DistMult, ComplEx), and for graph neural networks operating on heterogeneous graphs.

### 3.3 The pipeline this project implements

```
Raw sentences → Triplet extraction (3 tools) → Lemmatization → Evaluation (P/R/F1) → Best tool selection → Knowledge graph visualization
```

Each component is implemented as a standalone module, allowing easy substitution and re-evaluation.

---

## 4. Dataset — Custom-Collected Sentences from Three Domains

### 4.1 Why a custom dataset?

Each of the three tools was originally evaluated on a domain-specific corpus. To compare them fairly, a unified, multi-domain benchmark is required. The team curated **~300 sentences** distributed across three textual domains chosen to span the syntactic diversity of everyday written English:

| Domain | Source character | Expected challenges for triplet extraction |
|---|---|---|
| **News** | Newspapers / online articles | Long sentences, named entities, complex modifiers, embedded clauses |
| **Novel** | Novels and literary text | Indirect speech, figurative language, complex predicate structures |
| **Caption** | Image descriptions (likely COCO-style) | Short, syntactically simple, explicit subject-action-object structure |

Approximately **100 sentences per domain** were collected, with **80 used for evaluation** and the remaining 20 reserved for visualization (per the configuration `num_test = 80` in `evaluate_f1_score.py`).

### 4.2 Annotation protocol

Each sentence was manually annotated with one or more ground-truth triplets in a consistent text format:

```
subject1,relation1,object1;subject2,relation2,object2
```

- **Comma** separates the three components of a single triplet.
- **Semicolon** separates multiple triplets within a single sentence.
- **One sentence per line** in the annotation files.

A sentence with no extractable triplets is represented by an empty line. The annotations are stored in files named `<domain>_sentences-label.txt` (e.g., `news_sentences-label.txt`).

### 4.3 Annotation challenges and judgement calls

Manual triplet annotation for arbitrary natural language is genuinely difficult — there is no canonical "correct" decomposition for many sentences. Some recurring decisions the team had to make include:

- **How aggressively to expand multi-word entities.** *"The president of the United States"* could be a single subject node or three (president, United, States). The team's annotation appears to favor compact noun phrases.
- **How to handle modifiers.** *"the brown dog runs quickly"* could yield `<dog, runs, quickly>` or `<dog brown, runs, quickly>` or `<dog, runs quickly, _>`. The team's choice trades off granularity against canonical form.
- **How to split conjunctions.** *"Alice and Bob went to Paris"* might be one triplet `<Alice and Bob, went to, Paris>` or two `<Alice, went to, Paris>` and `<Bob, went to, Paris>`.

These choices are not theoretically unique and inject some annotator-specific noise into the ground truth — a reality the project acknowledges by adopting the loose-matching evaluation scheme (Section 8).

---

## 5. Tool 1 — spaCy with a Rule-Based Dependency Parser

**Notebook:** `project_spacy.ipynb`
**Approach:** Hand-crafted rules over spaCy's dependency parse tree.

### 5.1 Theoretical framing

spaCy is a production-grade NLP library that provides pre-trained statistical models for tokenization, POS tagging, dependency parsing, and named-entity recognition. The `en_core_web_sm` model used here is a small (~12 MB) English model trained on OntoNotes 5.

For each token in a sentence, spaCy assigns:
- **POS tag** (e.g., NOUN, VERB, PROPN)
- **Dependency label** (e.g., `nsubj` for nominal subject, `dobj` for direct object, `ROOT` for the syntactic head)

The team's strategy is to **handcraft rules over these labels** to identify subject, predicate, and object roles. This is the simplest possible triplet-extraction approach — and serves as a baseline against the more sophisticated learned tools.

### 5.2 Rule specification

The rule system, implemented in `processSubjectObjectPairs`, defines three categories of dependency labels:

```python
def isRelationCandidate(token):
    # Tokens that may form the predicate
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)

def isConstructionCandidate(token):
    # Tokens that modify a subject or object (e.g., compounds, prepositions)
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)
```

The matching logic uses substring containment (`in`), so e.g. `ROOT` matches the literal root token, and `subj` matches `nsubj`, `nsubjpass`, `csubj`, and `csubjpass` — a single rule covers all subject types.

### 5.3 The extraction algorithm

```python
def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''
    return (subject.strip(), relation.strip(), object.strip())
```

The algorithm walks tokens left-to-right, accumulating modifiers (compounds, prepositions) into a buffer that gets flushed to either the subject or object when a `subj`/`obj` token appears. The relation is built from the syntactic root and any adjectival/attributive material.

### 5.4 Strengths and weaknesses of this approach

**Strengths:**
- Fully transparent and debuggable — every triplet can be traced back to a deterministic rule firing on a specific dependency label.
- No training data required.
- Fast inference (depends only on spaCy's parse).

**Weaknesses:**
- **Single triplet per sentence.** The algorithm produces *one* `(subject, relation, object)` tuple per sentence, even when the sentence contains multiple distinct relations. This is a structural limit.
- **Aggressive accumulation.** Modifiers are appended to subject/object in linear order, sometimes producing nonsense like `London , 50-mile be major settlement , Thames east England head estuary Sea millennia` (visible in the notebook output) — a syntactically meaningful sentence becomes a meaningless triplet because the rules cannot identify which prepositional phrases attach to which entity.
- **Limited dependency-label coverage.** Sentences with non-standard structures (passives, gerunds, complex clauses) are mishandled because the rule set was hand-crafted for prototypical S-V-O English.

### 5.5 Demonstration on the London paragraph

Running on a multi-sentence paragraph about London produces the following triplets:

```
London , be capital large , England
London , 50-mile be major settlement , Thames east England head estuary Sea millennia
Londinium , found by , Romans
City core that , ancient square retain medieval , London miles − boundaries limits
City , be hold status , Westminster
London , govern by , Mayor London
London , locate , southeast England
Westminster , locate , London
London , be big city , Britain
```

The clean triplets — *Londinium found by Romans*, *London locate southeast England*, *Westminster locate London*, *London be big city Britain* — are interspersed with malformed ones where the rules over-accumulate modifiers. This is characteristic of the approach: high precision on simple sentences, low precision on complex ones.

---

## 6. Tool 2 — Stanford OpenIE

**Notebook:** `OpenIE_in_Python.ipynb`
**Approach:** Pre-trained, learned information-extraction system.

### 6.1 Theoretical framing

**Stanford OpenIE** (Angeli, Premkumar, Manning, ACL 2015) is an *open-domain* information extraction system — meaning it produces triplets without committing to a fixed schema of relation types. Its pipeline is:

1. **Dependency parsing.** Each sentence is parsed into its dependency tree.
2. **Tree traversal.** The dependency tree is recursively traversed to identify clauses (units of meaning that potentially contain a relation).
3. **Clause segmentation.** A logistic regression classifier, trained on supervised data, predicts which dependency arcs to keep or split when isolating a clause.
4. **Relation extraction.** From each isolated clause, a `<subject, relation, object>` triplet is read off.
5. **Entailment-based reduction.** The system can produce shorter "shortened" triplets that are *entailed* by the original sentence — useful for matching against terse ground-truth annotations.

A key strength of OpenIE is that it produces **multiple triplets per sentence** when warranted — unlike the spaCy rule system. This dramatically improves recall on multi-clause sentences.

### 6.2 Implementation

The notebook uses the `stanford_openie` Python package, which wraps the Stanford CoreNLP server:

```python
from openie import StanfordOpenIE
import numpy as np

# Load 100 sentences from a text file
sent_df = np.array([])
with open('100sentences_CW.txt') as infile:
    for line in infile:
        sent_df = np.append(sent_df, line)

# Extract triplets — one inner loop per sentence
df = np.array([])
with StanfordOpenIE() as client:
    for i, line in enumerate(sent_df[0:100]):
        text = line
        svo = ''
        for triple in client.annotate(text):
            s = triple['subject'] + ',' + triple['relation'] + ',' + triple['object'] + ';'
            svo += s
        df = np.append(df, svo)
```

The `StanfordOpenIE` context manager auto-launches a Java CoreNLP server (`java -Xmx8G -cp ...`) on port 9000. Each sentence is sent to the server, which returns a list of triplet dicts; the team concatenates them with semicolons and writes the result to disk.

### 6.3 Sample output

The notebook displays the first five sentences' predictions:

```
['They,stick,good;They,look,good']
['']
['They,are,super stylish;They,are,stylish;I,can share,them']
['These,make,my phone look so stylish']
["It,would,would 've nice;It,get,it"]
```

Note:
- **Multiple triplets per sentence.** Sentence 3 produces three triplets, capturing the conjunction explicitly.
- **Empty results allowed.** Sentence 2 produced no triplets; the system correctly returns nothing rather than fabricating.
- **Some edge cases.** Sentence 5 shows a less-clean extraction (`It would would 've nice`), reflecting the difficulty of contractions and informal text.

### 6.4 Output formatting

The team explicitly cleans up trailing semicolons:

```python
final_df = np.array([])
df = df.reshape(sent_df.shape[0], 1)
for line in df:
    if line[0] != '' and line[0][-1] == ';':
        final_df = np.append(final_df, line[0][0:len(line[0])-1])
    else:
        final_df = np.append(final_df, line[0])
final_df = final_df.reshape(sent_df.shape[0], 1)
np.savetxt('predicted_triplets.txt', final_df, fmt='%s')
```

This produces output in the same format as the ground-truth labels, ready for direct comparison.

### 6.5 Strengths and weaknesses

**Strengths:**
- Multi-triplet output captures complex sentences faithfully.
- Trained on real linguistic data; handles many English idioms gracefully.
- Built-in entailment reduction shortens overly-long triplets to canonical forms.

**Weaknesses:**
- **Heavy dependency.** Requires Java, the full Stanford CoreNLP package, and 8GB of RAM. Not lightweight.
- **Server startup overhead.** First-call latency is several seconds.
- **Tendency to over-generate.** Sometimes produces near-duplicate triplets (`<They, are, super stylish>` and `<They, are, stylish>`), inflating both precision and recall noise.

---

## 7. Tool 3 — Stanford Scene Graph Parser via SPICE

**Code:** `parse_sentence.py`, `create_coco_sg.py`, `process_spice_sg.py`
**Approach:** Domain-specific parser trained on image captions.

### 7.1 Theoretical framing

The **Stanford Scene Graph Parser** (Schuster, Krishna, Chang, Fei-Fei, Manning, 2015) is designed specifically for **image-grounded text** — captions, visual descriptions, and similar domains. It maps a sentence to a **scene graph**, a structured representation comprising:

- **Object nodes** (the entities in the described image)
- **Attribute annotations** (color, size, state)
- **Relation edges** (spatial and action relationships between objects)

The parser is wrapped inside **SPICE** (Semantic Propositional Image Caption Evaluation), an automatic metric for image captioning that compares generated captions to references via scene-graph overlap. SPICE has the side effect of producing a parsed scene graph for any input sentence — and the team exploits this by using SPICE as a black box that emits scene graphs, then post-processing those graphs into triplets.

### 7.2 Pipeline implementation

Three Python scripts implement the full pipeline:

#### `create_coco_sg.py` — Demonstration that SPICE-as-parser works

```python
sent_list = [
    'a man in tan shirt sitting at a table with food.',
    'a man who is sitting at a table with a plate of food in front of him.',
    'man sitting at a dinner table at an open restaurant.',
    'the man smiles over a table covered with plates of food.',
    'a man sitting at a table with an almost empty plate of food'
]

gts = {}
res = {}
for img_id, this_sent in enumerate(sent_list):
    gts[img_id] = []
    gts[img_id].append(this_sent)
    res[img_id] = []
    res[img_id].append('place holder')   # SPICE requires a candidate; we ignore the score

scorer = Spice()
score, scores = scorer.compute_score(gts, res)
```

The trick is that SPICE's `compute_score` parses each "ground-truth" caption (the actual sentences we want triplets for) into a scene graph and writes the result to `./spice/sg.json`. The "candidate" placeholders are necessary to satisfy SPICE's API but are otherwise unused.

#### `parse_sentence.py` — End-to-end driver

This script combines the SPICE-call with post-processing in one file:

```python
data_path = './data/caption_sentences.txt'
sent_list = [item for item in open(data_path, 'r')]

# 1. Parse with SPICE (writes sg.json)
gts = {img_id: [sent] for img_id, sent in enumerate(sent_list)}
res = {img_id: ['place holder'] for img_id in gts}
scorer = Spice()
score, scores = scorer.compute_score(gts, res)

# 2. Read the JSON, extract triplets, lemmatize
sg_json = json.load(open('./spice/sg.json'))
all_triplets = []
for img_id in img_ids:
    sg_temp = sg_json[str(img_id)]
    rela = sg_temp['rela']     # list of relation strings
    sbj = sg_temp['subject']   # list of subject strings
    obj = sg_temp['object']    # list of object strings

    this_triplet = []
    for i in range(len(rela)):
        rela_temp = lemmatizer(rela[i].strip().lower())
        sbj_temp  = lemmatizer(sbj[i].strip().lower())
        obj_temp  = lemmatizer(obj[i].strip().lower())
        this_triplet.append([sbj_temp, rela_temp, obj_temp])
    all_triplets.append(this_triplet)

# 3. Write to text file in the standard format
with open('extracted_triplets.txt', 'w') as f:
    for sen_i, triplets in enumerate(all_triplets):
        for trip_i, trip in enumerate(triplets):
            sep = '' if trip_i == 0 else ';'
            f.write(sep + trip[0] + ',' + trip[1] + ',' + trip[2])
        f.write('\n')
```

The lemmatizer is NLTK's `WordNetLemmatizer` wrapped in a fallback that tries verb-form lemmatization if noun-form returns the original word — a small but useful detail for handling words like *running* (verb) vs. *running* (noun).

#### `process_spice_sg.py` — Vocabulary management

This auxiliary script (adapted from the SGAE repository) builds a unified word-to-index dictionary across the COCO scene graph training and validation splits, persisting it for use by downstream models. While not directly part of the triplet evaluation, it documents the team's interaction with the upstream COCO-trained scene-graph infrastructure.

### 7.3 Why this design is methodologically interesting

Using SPICE-as-parser is an elegant hack: the team didn't need to train a scene-graph parser from scratch (which would require MS COCO captions and a lot of GPU time), nor did they need to integrate Stanford's standalone scene-graph parser via Java. Instead, they exploit the fact that an existing automated metric *internally* parses sentences as a side effect of its operation.

The cost is that SPICE's parse output is biased toward the kind of language seen in image captions — short, concrete, present-tense — which becomes apparent in the domain-stratified results.

### 7.4 Strengths and weaknesses

**Strengths:**
- Excellent on captions by construction.
- Produces **attribute annotations** in addition to triplets — richer than OpenIE (though attributes are not used in the F1 evaluation).
- Multi-triplet output per sentence.

**Weaknesses:**
- **Severe domain transfer failure.** Trained on visual-grounding data, struggles badly on news and especially novel text where syntax and vocabulary diverge.
- **Heavy infrastructure.** Requires the SPICE Java jar plus its dependencies.
- **Indirect API.** Using SPICE as a parser is a workaround, not the official intended use.

---

## 8. Evaluation Methodology — Precision, Recall, F1 with Lemmatized Matching

**Code:** `evaluate_f1_score.py`

### 8.1 Why simple string matching is insufficient

A naive evaluation would compare predicted and ground-truth triplets as raw strings. This fails immediately on trivial morphological variation:

- Predicted: `<dog, runs, park>`
- Ground truth: `<dogs, running, parks>`

The two are semantically identical but a string match returns 0. The team addresses this by **lemmatizing every word in every triplet** before comparison.

### 8.2 Lemmatization function

```python
from nltk.stem import WordNetLemmatizer
from functools import partial

def change_word(lem, word_ori):
    """Lemmatize, with verb-form fallback."""
    word_ori = word_ori.lower()
    word_change = lem.lemmatize(word_ori)
    if word_change == word_ori:
        word_change = lem.lemmatize(word_ori, 'v')
    return word_change

lem = WordNetLemmatizer()
lemmatizer = partial(change_word, lem)
```

The fallback handles words like `running`: `lemmatize('running')` returns `'running'` (treated as noun), but `lemmatize('running', 'v')` returns `'run'`. Without the fallback, action verbs would be inconsistently normalized.

For multi-word subjects/objects, each word is lemmatized independently and rejoined with spaces.

### 8.3 Two matching schemes

The team uses two matching schemes that report increasingly lenient results:

#### Scheme 1: Exact lemma matching

A predicted triplet matches a ground-truth triplet only if all three components are *exactly* equal after lemmatization:

```python
if (this_pred_trip[0] == this_label_trip[0] and
    this_pred_trip[1] == this_label_trip[1] and
    this_pred_trip[2] == this_label_trip[2]):
    pred_label_mtx[i, j] = 1
```

#### Scheme 2: Loose lemma matching (substring overlap)

If exact match fails, a softer condition is checked: any word in the predicted component must overlap (as a substring, in either direction) with any word in the corresponding ground-truth component:

```python
def wd_list_match(list1, list2):
    num_match = len([wdd for wd in list1 for wdd in list2
                     if wdd in wd or wd in wdd])
    return num_match != 0

# Loose match logic:
if (wd_list_match(pred_subj_wd, label_subj_wd) and
    wd_list_match(pred_rel_wd, label_rel_wd) and
    wd_list_match(pred_obj_wd, label_obj_wd)):
    pred_label_mtx[i, j] = 1
```

A predicted `<dog brown, run, park large>` would match ground-truth `<dog, run, park>` because `dog ∈ "dog brown"`, `run = run`, and `park ∈ "park large"`.

### 8.4 Why both schemes matter

The two schemes capture different aspects of extractor quality:

- **Exact matching** rewards precise, canonical triplet output. A tool that consistently produces minimal triplets (just the head noun, just the verb stem) does well here.
- **Loose matching** rewards capturing the *correct entities and relations* even with extra modifier words. A tool that produces more verbose but semantically correct triplets does well here.

For practical knowledge-graph construction, **loose matching is arguably more meaningful** — downstream entity-resolution and graph-merge steps will normalize modifier variants regardless. But exact matching is harder to game and gives a sharper signal.

### 8.5 Aggregation: micro-averaged precision, recall, F1

The team uses **micro-averaging across sentences**: total counts of matched/unmatched predictions and labels are summed, then precision and recall are computed once at the corpus level:

```python
precision = np.sum(num_matched_pred) / float(np.sum(num_pred_lst))
recall   = np.sum(num_matched_label) / float(np.sum(num_label_lst))
f1_score = 2 * (precision * recall) / (precision + recall)
```

Where:
- `num_matched_pred[i]` = number of predicted triplets in sentence `i` that match *some* label triplet
- `num_matched_label[i]` = number of label triplets in sentence `i` that are matched by *some* prediction
- `num_pred_lst[i]` = total predicted triplets in sentence `i`
- `num_label_lst[i]` = total label triplets in sentence `i`

The asymmetry of "any-match" pairing is intentional: a single prediction matching multiple labels (common with loose matching) counts as one matched prediction *and* multiple matched labels — capturing both tools' tendency to coalesce details and the truth's tendency to over-decompose.

### 8.6 Test set size

The script reserves **80 sentences out of 100 per domain** for evaluation:

```python
num_test = 80
eval_ind = list(np.arange(100)[:num_test])
```

The remaining 20 sentences are used for visualization (Section 11) — a deliberate train/test-style separation, even though no model fitting is performed. This protects against cherry-picking visualization examples.

---

## 9. Results

### 9.1 Exact-matching results

Reported as `precision / recall / F1` (all in %):

| Method | News | Novel | Caption |
|---|---|---|---|
| spaCy | 18.60 / 16.49 / **17.49** | 19.53 / 17.01 / **18.18** | 37.68 / 32.30 / **34.78** |
| OpenIE | 8.02 / 19.59 / **11.38** | 23.81 / 47.62 / **31.75** | 19.87 / 19.25 / **19.56** |
| SPICE | 13.08 / 14.43 / **13.73** | 1.50 / 1.36 / **1.43** | 34.51 / 30.43 / **32.34** |

### 9.2 Loose-matching results (exact + substring overlap)

| Method | News | Novel | Caption |
|---|---|---|---|
| spaCy | 40.70 / 36.08 / **38.25** | 40.70 / 36.08 / **38.25** | 57.25 / 50.31 / **53.55** |
| OpenIE | 59.49 / 52.58 / **55.82** | 59.49 / 52.58 / **55.82** | 83.33 / 49.07 / **61.77** |
| SPICE | 28.97 / 29.90 / **29.43** | 26.32 / 22.45 / **24.23** | 52.82 / 47.20 / **49.85** |

### 9.3 Per-domain best tool

| Domain | Best (Exact F1) | Best (Loose F1) |
|---|---|---|
| **News** | spaCy (17.49) | **OpenIE (55.82)** |
| **Novel** | OpenIE (31.75) | **OpenIE (55.82)** |
| **Caption** | spaCy (34.78) | **OpenIE (61.77)** |

OpenIE wins on every domain under loose matching, and on novels under exact matching. spaCy wins on news and captions under exact matching — a result driven by spaCy's tendency to produce shorter, more canonical triplets that match exactly more often.

### 9.4 Domain difficulty ranking

Across all tools, F1 is consistently highest on captions, intermediate on news, and lowest on novels (by SPICE; by OpenIE/spaCy news and novels are roughly tied). This ordering reflects intuitive textual difficulty:

- **Captions**: short, syntactically simple, high content-word density, explicit `<entity, action, entity>` form.
- **News**: longer sentences, more named entities, more dependent clauses, but still factual register.
- **Novels**: figurative language, indirect discourse, complex predicates, abstract relationships — genuinely hard for any rule-based or shallow learned system.

### 9.5 The dramatic SPICE-on-novels result (1.43% exact F1)

The most striking single number in the results table is **SPICE's 1.43% exact F1 on novels** — essentially zero. This is a clean illustration of catastrophic domain transfer failure: a parser trained on `<man, sit at, table>` style captions cannot meaningfully process *"He was a tall, gaunt man, with a most lugubrious expression"* and similar novelistic prose. Even loose matching only recovers SPICE on novels to 24.23%, still well below the other tools.

This finding is methodologically important: it confirms that **off-the-shelf NLP tools should never be applied blindly across domains**, and validates the project's motivation of running a controlled cross-domain comparison.

### 9.6 The exact-matching anomaly: OpenIE's low precision on news (8%)

OpenIE's exact-matching precision on news (8.02%) is surprisingly low — much lower than its recall (19.59%). The cause is OpenIE's tendency to emit *multiple variants* of the same logical triplet (e.g., `<It, would, would 've nice>` and `<It, get, it>` from the same sentence) and *long, modifier-heavy* triplets that fail exact matching against compact ground-truth annotations.

Under loose matching, this changes dramatically: OpenIE's precision on news jumps to 59.49% — a 7.4× increase — because the substring-overlap rule forgives modifier variation. This is the clearest demonstration in the results that **the match scheme dominates the apparent quality of OpenIE's output**, while spaCy and SPICE are less sensitive.

---

## 10. Comparative Analysis and Discussion

### 10.1 Why OpenIE wins overall

Three factors combine to make OpenIE the best general-purpose tool:

1. **Multi-triplet output.** Unlike spaCy's single-triplet-per-sentence rule system, OpenIE decomposes complex sentences into multiple triplets, dramatically improving recall on multi-clause input.

2. **Trained on broad corpora.** OpenIE's logistic regression classifier was trained on diverse, manually-annotated news and Wikipedia text. This gives it broader coverage than the caption-specific Scene Graph Parser.

3. **Entailment-aware design.** OpenIE explicitly produces *both* the long-form triplets and entailed shorter forms. The shorter forms tend to match human annotations.

The cost is computational: OpenIE requires a Java server and 8GB of RAM, making it unsuitable for low-resource deployments where spaCy's pure-Python rule system would be the practical choice.

### 10.2 Why spaCy is competitive despite its simplicity

spaCy's rule-based approach achieves 38.25% loose F1 on news/novel and 53.55% on captions — surprisingly competitive given that the entire system is ~50 lines of Python rules. Two reasons:

1. **spaCy's pre-trained dependency parser is very good.** The hard part — POS tagging and dependency tree construction — is solved by `en_core_web_sm`. The rules just read off the result.
2. **The captioning sweet spot.** spaCy's single-triplet-per-sentence design happens to align with the caption domain, where most sentences encode a single dominant relation.

The takeaway: **for many use cases, a high-quality dependency parser plus simple rules captures most of the value**. The diminishing-returns gap between spaCy (38%) and OpenIE (56%) loose-F1 may not justify OpenIE's heavier infrastructure for resource-constrained applications.

### 10.3 SPICE — a tool out of its element

The Scene Graph Parser is the right tool for one specific job (parsing image captions for visual grounding) and the wrong tool for general triplet extraction. Its failure modes on novels and news are not bugs but consequences of training distribution: it has never seen sentences like *"He was wearing his trademark frown"* or *"The Senate passed the bill 51-49"* and lacks the vocabulary or syntactic heuristics to handle them.

This is a useful negative result. Domain-specific tools, even well-engineered ones, do not transfer.

### 10.4 The match-scheme sensitivity question

Every tool's F1 jumps substantially from exact to loose matching. The magnitudes are revealing:

| Tool | Avg Exact F1 | Avg Loose F1 | Multiplier |
|---|---|---|---|
| spaCy | ~23% | ~43% | 1.9× |
| OpenIE | ~21% | ~58% | **2.7×** |
| SPICE | ~16% | ~35% | 2.2× |

OpenIE's larger multiplier confirms it produces more verbose, modifier-heavy triplets — they're often correct in spirit but fail exact match. spaCy's smaller multiplier confirms it produces compact triplets that either match exactly or miss entirely.

For practitioners: **if downstream graph construction will normalize entities anyway, loose-match F1 is the more relevant metric.** If downstream consumption requires exact lexical match (e.g., direct insertion into a strict schema), exact-match F1 governs.

### 10.5 The single-triplet-per-sentence ceiling for spaCy

A structural observation: the spaCy rule system produces *exactly one* triplet per sentence (or none, if no `subj` token is found). This is a hard upper bound on its recall: any sentence with multiple ground-truth triplets cannot reach 100% recall regardless of rule quality. To break this ceiling, the rules would need a clause-segmentation step — essentially reinventing OpenIE's approach.

The team's choice not to extend the rules in this direction is reasonable: at some point, hand-crafted rules become prohibitively complex, and at that point one should switch to a learned system. The project's results document the natural breakpoint.

### 10.6 What this benchmark does and doesn't measure

**It does measure:** the practical out-of-the-box quality of three popular triplet extractors on three real text domains, evaluated by a consistent, lemma-aware protocol with both strict and lenient matching.

**It does not measure:**
- **Computational cost.** OpenIE is ~100× slower than spaCy in wall-clock time per sentence. The trade-off matters for production.
- **Multi-sentence document handling.** All three tools operate sentence-by-sentence; cross-sentence anaphora resolution is not in scope.
- **Knowledge graph quality after merging.** Two extractors with the same triplet F1 might produce graphs of very different quality after entity resolution and merging — but that downstream step is not evaluated here.
- **Modern transformer baselines.** Tools like REBEL (Cabot & Navigli, 2021) or LLM-based extractors (post-2022) substantially outperform all three of these tools on this kind of task. Their absence from the benchmark reflects when the project was conducted, not a methodological gap.

---

## 11. Visualization — Triplets to Knowledge Graphs

**Code:** `main.py`

The team implements a simple visualization pipeline that takes any triplets file in the standard format and renders it as a directed graph using NetworkX and matplotlib:

```python
import networkx as nx
import matplotlib.pyplot as plt

def get_triples(file_name):
    triples = []
    file = open(file_name, 'r')
    for lines in file:
        new_line = lines.rstrip()
        tri_list = new_line.split(';')         # split sentences with multiple triples
        for triple in tri_list:
            single_triple = triple.split(',')  # split into [s, r, o]
            triples.append(single_triple)
    return triples

def printGraph(triples):
    G = nx.DiGraph()
    for triple in triples:
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edges_from([(triple[0], triple[1]), (triple[1], triple[2])])

    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw_networkx(G, pos, edge_color='black', width=1, linewidths=1,
                     node_size=500, font_size=8, node_color='seagreen',
                     alpha=0.8, arrows=True, arrowstyle='-|>', arrowsize=10,
                     labels={node: node for node in G.nodes()})
    plt.axis('off')
    plt.show()
```

Notable design choices:

1. **DiGraph (directed).** The graph is directed because relations are asymmetric: `<London, capital of, England>` is not the same as `<England, capital of, London>`.
2. **Predicate-as-node, not edge-label.** The relation token becomes a *node* with edges `(subject → relation)` and `(relation → object)`, rather than the more standard `(subject → object)` with `relation` as the edge label. This is a slightly unusual choice — it produces a tripartite-ish layout and visualizes the relation tokens as first-class entities. The standard alternative would be `G.add_edge(triple[0], triple[2], label=triple[1])`.
3. **Spring layout.** Force-directed layout, the standard default. Works adequately for graphs of dozens to a few hundred nodes.

The visualization pipeline operates on the **20 sentences per domain reserved for visualization** (the complement of the evaluation set). Running it with the best-tool predictions for each domain gives the team's final knowledge-graph artifacts.

---

## 12. Conclusions and Future Work

### 12.1 Conclusions

This project produces three principal findings:

1. **Stanford OpenIE is the best general-purpose triplet extractor among the three tools tested**, achieving 55.82% F1 on news and novel domains and 61.77% on captions under loose lemma matching. Its multi-triplet-per-sentence design and broad training corpus give it an advantage over both the simpler spaCy rules and the domain-specific Scene Graph Parser.

2. **spaCy with hand-crafted rules is a competitive lightweight alternative**, especially on captions (53.55% F1). For applications where Java/CoreNLP infrastructure is impractical, a well-designed dependency-parse rule system captures most of the value at a fraction of the cost.

3. **The Stanford Scene Graph Parser does not transfer outside its training domain.** On novels its loose-match F1 is 24.23% — barely better than chance — while on captions it reaches 49.85%. This 26-point spread is the cleanest empirical illustration in the project of **the limits of off-the-shelf NLP tooling under domain shift**.

A subsidiary methodological contribution is the **dual matching scheme** (exact + loose lemma matching with substring overlap). The factor-of-2-to-3 difference between the two schemes' F1 figures shows that the choice of evaluation protocol can dominate apparent tool quality, and any benchmark of triplet extractors should report both.

### 12.2 Future Work — Acknowledged in the Report

The team identifies several directions:

1. **Extending evaluation to the full 100 sentences per domain.** The current 80/20 split was chosen for visualization purposes; with no test/visualization split needed, all 100 could feed the F1 computation.
2. **Including subject- and object-level F1 as separate metrics**, beyond whole-triplet matching. This would help diagnose *which component* each tool struggles with.
3. **Analyzing failure modes per tool.** Specifically, identifying the linguistic constructions where each tool consistently fails — a qualitative complement to the quantitative F1 numbers.

### 12.3 Future Work — Not in the Report But Implied

Several extensions would meaningfully improve the benchmark:

4. **Add a modern transformer baseline.** REBEL, OpenIE6, or an instruction-tuned LLM (GPT-4, Claude, Llama) prompted as a triplet extractor would likely outperform all three tools tested. Including such baselines would contextualize the gap between classical NLP tooling and modern systems.

5. **Add a confidence-weighted F1.** OpenIE provides a confidence score per triplet; ranking and thresholding by confidence enables precision-recall curves rather than single F1 numbers.

6. **Compute inter-annotator agreement on the ground-truth labels.** Triplet annotation is genuinely ambiguous; reporting Cohen's κ across two annotators would quantify the noise floor against which the tools' F1 should be interpreted.

7. **Extend to entity resolution and graph merging.** The current evaluation stops at triplet extraction. The downstream step — merging triplets across sentences into a unified knowledge graph — has its own quality measures (entity coverage, relation density, graph connectivity) and should be benchmarked separately.

8. **Increase domain diversity.** Three domains is a useful start; expanding to scientific abstracts, legal text, social media posts, and dialogue would make the benchmark more representative of real-world deployment scenarios.

9. **Standardize the SPICE-on-novel failure into a named negative result.** The 1.43% exact-F1 finding is strong evidence of training-distribution dependency that deserves a more rigorous error analysis — what specific syntactic constructions break the parser?

---

## 13. Appendix — File Inventory

| File | Type | Role |
|---|---|---|
| `STAT_451_Project_Proposal.pdf` | PDF | Original project proposal (3 pages); defines goals, methodology, evaluation plan |
| `Stat_451_Project_Slide.pptx` | PowerPoint | Final presentation slide deck (12 slides) |
| `README.md` (root) | Markdown | Project overview, folder structure, tool list, performance tables |
| `project_spacy.ipynb` | Notebook | spaCy rule-based triplet extractor implementation |
| `OpenIE_in_Python.ipynb` | Notebook | Stanford OpenIE wrapper |
| `parse_sentence.py` | Script | SPICE-as-parser pipeline: parse → extract → write |
| `create_coco_sg.py` | Script | Demonstration of SPICE parsing on a small caption set |
| `process_spice_sg.py` | Script | Vocabulary management for COCO scene graphs (auxiliary) |
| `evaluate_f1_score.py` | Script | F1 evaluation with exact + loose lemma matching |
| `main.py` | Script | Knowledge-graph visualization (NetworkX + matplotlib) |

**Folder structure (per project README):**

| Folder | Contents |
|---|---|
| `code/` | Tool-specific implementations |
| `data/` | Annotated sentences (`<domain>_sentences-label.txt`) |
| `predictions/` | Tool outputs (`predicted_triplets_<domain>_<tool>.txt`) |
| `evaluation/` | F1 evaluation script |
| `visualization/` | Graph rendering code |
| `slides and reports/` | Proposal, slides, final write-up |

**External dependencies:**

- **spaCy** with `en_core_web_sm` model
- **stanford_openie** Python wrapper + Stanford CoreNLP 2018-10-05 (Java, 8GB RAM)
- **SPICE** (Java) and its associated dependencies for scene-graph parsing
- **NLTK** for `WordNetLemmatizer`
- **NetworkX** + **matplotlib** for visualization

---

*End of report.*
