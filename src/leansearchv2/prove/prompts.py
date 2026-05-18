"""Prompts used by the prove task. Identical to the formulation used to
produce the paper's Table 3 numbers.

Four prompts:
- PROVER_INIT: theorem + retrieved docs -> initial proof attempt.
- PROVER_REFLECT: error + retrieved docs -> revised proof attempt.
- GET_QUERY_INIT: theorem -> retrieval queries for the initial attempt.
- GET_QUERY_REFLECT: theorem + failed proof + errors -> retrieval queries
  for the next reflection round.
"""

from __future__ import annotations


PROVER_INIT = """\
You are an expert at using Lean for formal proofs. Your task is to complete the following Lean 4 code. You have access to the following relevant theorems and definitions found through search.

## Lean 4 Code to Complete

```lean4
{lean_code}
```
## Relevant Search Results

{search_results}

## Critical Guidance

- Pay special attention to the search results above, as they contain relevant lemmas and theorems that may be directly applicable to this proof. Consider how to incorporate these results into your proof strategy.

## Final Requirement on the Formal Statement

Note that the given formal statement has been verified by Lean 4 and professional mathematicians, so you SHOULD NOT CHANGE the formal statement. Instead, use the exact code in the formal statement as your prefix.

### Your Response Format

Before producing the Lean 4 code to formally prove the given theorem, provide:
1. **Detailed proof plan:** Outline the main proof steps and strategies. The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof. Analyze how the relevant search results can be utilized in your proof.
2. **Final Lean 4 Code:** Provide the complete Lean 4 code block that includes the formal statement and your proof.

Your final code **MUST** begin with the exact, character-for-character original formal statement and follow the format:
```lean4
{lean_code}
```
"""


PROVER_REFLECT = """\
You are an expert at using Lean for formal proofs. Your task is to revise the following Lean 4 code. You will be provided with an incorrect proof, its error messages, and search results that contain relevant theorems and definitions.

## Incorrect Proof

```lean4
{proof}
```

## Error Messages

{error_msg}

## Relevant Search Results

{search_results}

## Critical Guidance on using External Information

**When the Compiler Reports Errors:**
If the `Error messages` section contains errors like:
- `module '...' does not exist`: You are trying to import a module (.lean file) that is not available in the current Mathlib version. Please use the **most safe import** `import Mathlib` to replace all your incorrect customized imports, or just use the original `import` in the formal statement (`import Mathlib` for most cases).
- `unknown tactic '...'`: This means the tactic you are trying to use is not recognized by Lean 4, likely because it does not exist in the current library or version you are using.
- `unknown identifier '...'` (or `Unknown constant`, `invalid field`): This mostly indicates that the identifier (which could be a theorem, definition, or variable) is not defined or imported in your current context. Please refer to the search results to find an appropriate replacement. If you cannot find any proper replacement in the search results, it indicates that such identifier is non-existent in Mathlib, please do not guess and **prove it from scratch** by `have`, `obtain`, or `show`, `suffices` tactics using known theorems from search results.
- `no goals to be solved`: it's possible that the current line of Tactic is redundant and can be deleted.

**Your Action Plan (MANDATORY):**
- You **MUST NOT** attempt to use the same non-existent item again.
- You **MUST** treat the `Relevant search results` as your **sole source of truth**.
- Your entire revised strategy **MUST** be built using **only** the tools and theorems found in the search results.

## Final Requirement on the Formal Statement

This is the most important rule. The formal statement of the theorem is **ABSOLUTELY IMMUTABLE**. It is a fixed prefix for your code.
- **DO NOT** change variable names, reorder hypotheses, change types, or change definitions.
- **All errors are guaranteed to be in the proof, never in the statement itself.** Your task is to fix the proof *within the confines* of the given statement.

## Your Response Format

Before producing the final code, you MUST provide a step-by-step thinking process:
1.  **Error Analysis:** Briefly explain what the error messages mean.
2.  **Revised Strategy:** Outline your new proof plan. Explain how you will use the search results to fix the errors.
3.  **Final Code:** Provide the complete, corrected Lean 4 code block.

Your final code **MUST** begin with the exact, character-for-character original formal statement and follow the format:
```lean4
{lean_code}
```
"""


GET_QUERY_INIT = """\
You are an expert in formal mathematics and Lean 4. Given a theorem statement and its proof context, your task is to generate {num_queries} precise search queries that would help find relevant lemmas, theorems, or definitions in a mathematical knowledge base.

**Theorem Statement:**
```lean4
{lean_code}
```

**Informal Description:**
{informal_statement}

Please first analyze the theorem and provide a natural language proof step by step, then generate exactly {num_queries} search queries that would be most helpful for proving this theorem. Avoid Proposing simple theorem names like "MonoidHom.map_one" alone. LeanSearch works best with natural language matching. Therefore, you should propose queries that include both the theorem name and a natural language description of its content.
Tips: when you need to search for definitions, you should clearly state that you are searching for a definition of something in the query, because LeanSearch are not sensitive to definitions.

Focus on:
1. Key mathematical concepts and objects involved in the proof
2. Main techniques or lemmas that might be needed
3. Specific properties or structures that appear in the statement

Return your response in the following JSON format. Please wrap the JSON block with ```json\\n ... \\n```:
```json
{{
    "queries": [
        "query 1 description",
        "query 2 description",
        "query 3 description",
        ...,
        "query {num_queries} description"
    ]
}}
```

Each query should be a clear, specific mathematical concept or theorem name that would likely appear in a mathematical database.
"""


GET_QUERY_REFLECT = """\
You are an expert in formal mathematics and Lean 4. Given a theorem statement, its proof context, and error information from a previous proof attempt, your task is to generate {num_queries} precise search queries that would help find relevant lemmas, theorems, or definitions to fix the errors.

**Theorem Statement:**
```lean4
{lean_code}
```

**Informal Description:**
{informal_statement}

**Previous Proof Attempt:**
```lean4
{proof}
```

**Error Messages:**
{error_msg}

### IMPORTANT: How to Handle "Not Found" Errors
If the error messages contain phrases like `Unknown constant ...`, `unknown identifier '...'` (or `Unknown constant`, `invalid field`), this is a critical signal. It means that an import path or a specific theorem name used in the previous proof is **invalid or non-existent in the current library**.

In this situation, your primary task is to find an equivalent or an alternative. One of your new search queries **MUST** be a **pure natural language description** of the *mathematical concept* you were trying to use.

**Example:**
- **IF** the error is: `Unknown constant 'Nat.prime_factor_unique'`
- **DO NOT** generate a query like: `"Nat.prime_factor_unique"` (this will fail again)
- **INSTEAD**, generate a query like: `"unique prime factorization theorem for natural numbers"` (this will help find the correct, existing theorem)

Based on the errors encountered, please analyze what went wrong and then generate exactly {num_queries} **New** search queries that would help find the missing pieces to fix these specific errors. Tips: when you need to search for definitions, you should clearly state that you are searching for a definition of something in the query, because LeanSearch are not sensitive to definitions.

Focus on:
1. Lemmas or theorems that could resolve the specific error messages, especially those related to incorrect theorem names or terms (`unknown identifier '...'` (or `Unknown constant '...'`, `invalid field '...'`)).
2. Alternative proof techniques that might avoid the current issues

Return your response in the following JSON format. Please wrap the JSON block with ```json\\n ... \\n```:
```json
{{
    "queries": [
        "query 1 description",
        "query 2 description",
        "query 3 description",
        ...,
        "query {num_queries} description"
    ]
}}
```

Each query should target specific mathematical concepts that could directly address the encountered errors.
"""
