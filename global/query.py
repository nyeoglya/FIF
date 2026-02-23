DOCUMENT_SUMMARIZATION_QUERY = r"""[SYSTEM ROLE]
You are an Expert Information Architect specializing in high-density knowledge synthesis. Your goal is to distill complex, multi-layered wiki documents into a singular, powerhouse summary that maintains 100% logic retention. The main task is to analyze the provided document and synthesize its entire scope into a Single, Unified Paragraph.

[CONSTRAINTS]
- Output must be exactly one continuous paragraph. Do not use line breaks or bullet points.
- You must integrate quantitative insights from tables and visual conclusions derived from image captions. These are not optional; they must ground your summary.
- Explicitly bridge the gap between the foundational concepts introduced at the beginning and the terminal conclusions/implications at the end.
- Eliminate meta-discourse (e.g., "This document states," "It is important to note"). Use precise terminology and active verbs to maximize facts-per-sentence.
- Limit the output to a maximum of 300 words.
- Ensure all critical key dates and primary proper nouns are preserved for indexing purposes
- Use only English when summarizing paragraph.

[DOCUMENT FOR ANALYSIS]
"""

IMAGE_OCR_QUERY = "Extract all text from the image. Maintain the original structure. If no text is detected, return an empty string ('') only. No introduction or closing remarks."
IMAGE_EXPLANATION_QUERY = "Provide a concise summary of this image in 2-3 sentences. Focus on the core subject, the setting, and the most striking visual element. Avoid filler words; be direct and precise."
MODALITY_AGNOSTIC_QUERY = "Retrieve passage, table or image (maybe with a caption) that provides an answer to the given query."
MODALITY_INSTRUCTION_QUERY = {
    "text": "Given a question, retrieve text passages that answer the question",
    "table": "Given a question, retrieve table-format texts that answer the question. A table can include an image within a cell.",
    "image": "Given a question, retrieve image-description (or OCR) pairs that answer the question"
}

SUBQUERY_DIVIDE_QUERY = """You are a retrieval-oriented **query decomposer**.

Goal - Produce the smallest set (1 - 5) of **component-targeting sub-queries**.  
Each sub-query must describe **one retrievable component** (sentence, paragraph, table row, figure, etc.) whose embedding should be matched.  
Together, the sub-queries must supply all the information needed to answer the original question.

Guidelines  
1. **Entity & noun-phrase coverage** Every noun phrase and named entity that appears in the original question must appear **at least once across the entire set** of sub-queries (you may distribute them). Keep each phrase exactly as written.  
2. **One-component rule** A sub-query should reference only the facts expected to co-occur **within the same component**. If two facts will likely be in different components, put them in different sub-queries.  
3. **No unnecessary splitting** If the whole answer can be found in a single component, return only one sub-query.  
4. **De-contextualize** Rewrite pronouns and implicit references so every sub-query is understandable on its own.  
5. **Keyword distribution** Spread constraints logically (e.g., one sub-query for “light rail completion date”, another for “city with a large arched bridge from the 1997 Australia rugby-union test match”).  
6. **Remove redundancy** Merge duplicate or paraphrased sub-queries before you output.  
7. **Ordering for dependencies** If the answer to one sub-query is needed for another, place the prerequisite first.  
8. **Output format**: Return the sub-queries separated by a semicolon (;). Do not provide any keys, explanations, introduction, or extra text. Ensure each sub-query is on a new line for clarity.

Question: {query}

Output: """

SUBQUERY_MODALITY_QUERY = """Instruction: You are a modality selector for multimodal QA.

Task: Given the single sub-question below, choose the one modality that is most appropriate for obtaining its answer.

Allowed modalities:
* text: unstructured prose (paragraphs, sentences, propositions)
* table: structured rows/columns (spreadsheets, stats tables, infoboxes)
* image: visual information (photos, posters, logos, charts)

Heuristics:
1. Numeric totals, percentages, year-by-year figures → table
2. Visual appearance, colours, logos, “what does . . . look like” → image
3. Definitions, roles, biographies, causal explanations, quotes → text
4. If two modalities could work, pick the one that will yield the answer fastest.

Output format: Return only the modality label on a single line – exactly text, table, or image. No JSON, no additional text.

Subquery: {subquery}
Output: """

DEMONSTRATION_PROMPT = """
/*
[Table]
Title: Gonzalo Higuain
Section: Career statistics | Club

Club | Season | League - Division | League - Apps | League - Goals | National Cup - Apps | National Cup - Goals | League Cup - Apps | League Cup - Goals | Continental - Apps | Continental - Goals | Other - Apps | Other - Goals | Total - Apps | Total - Goals 
 River Plate | 2004–05 | Argentine Primera División | 4 | 0 | 0 | 0 | — | — | 0 | 0 | — | — | 4 | 0
River Plate | 2005–06 | Argentine Primera División | 14 | 5 | 0 | 0 | — | — | 4 | 2 | — | — | 18 | 7
River Plate | 2006–07 | Argentine Primera División | 17 | 8 | 0 | 0 | — | — | 2 | 0 | — | — | 19 | 8
River Plate | Total | Total | 35 | 13 | 0 | 0 | — | — | 6 | 2 | 0 | 0 | 41 | 15
Real Madrid | 2006–07 | La Liga | 19 | 2 | 2 | 0 | — | — | 2 | 0 | — | — | 23 | 2
Real Madrid | 2007–08 | La Liga | 25 | 8 | 4 | 1 | — | — | 5 | 0 | — | — | 34 | 9
Real Madrid | 2008–09 | La Liga | 34 | 22 | 2 | 1 | — | — | 7 | 0 | 1 | 1 | 44 | 24
Real Madrid | 2009–10 | La Liga | 32 | 27 | 1 | 0 | — | — | 7 | 2 | — | — | 40 | 29
Real Madrid | 2010–11 | La Liga | 17 | 10 | 2 | 1 | — | — | 6 | 2 | — | — | 25 | 13
Real Madrid | 2011–12 | La Liga | 35 | 22 | 5 | 1 | — | — | 12 | 3 | 2 | 0 | 54 | 26
Real Madrid | 2012–13 | La Liga | 28 | 16 | 5 | 0 | — | — | 9 | 1 | 2 | 1 | 44 | 18
Real Madrid | Total | Total | 190 | 107 | 21 | 4 | — | — | 48 | 8 | 5 | 2 | 264 | 121
Napoli | 2013–14 | Serie A | 32 | 17 | 5 | 2 | — | — | 9 | 5 | — | — | 46 | 24
Napoli | 2014–15 | Serie A | 37 | 18 | 4 | 1 | — | — | 16 | 8 | 1 | 2 | 58 | 29
Napoli | 2015–16 | Serie A | 35 | 36 | 2 | 0 | — | — | 5 | 2 | — | — | 42 | 38
Napoli | Total | Total | 104 | 71 | 11 | 3 | — | — | 30 | 15 | 1 | 2 | 146 | 91
Juventus | 2016–17 | Serie A | 38 | 24 | 4 | 3 | — | — | 12 | 5 | 1 | 0 | 55 | 32
Juventus | 2017–18 | Serie A | 35 | 16 | 4 | 2 | — | — | 10 | 5 | 1 | 0 | 50 | 23
Juventus | 2019–20 | Serie A | 15 | 4 | 0 | 0 | — | — | 5 | 2 | 1 | 0 | 21 | 6
Juventus | Total | Total | 88 | 44 | 8 | 5 | — | — | 27 | 12 | 3 | 0 | 126 | 61
Milan (loan) | 2018–19 | Serie A | 15 | 6 | 1 | 0 | — | — | 5 | 2 | 1 | 0 | 22 | 8
Chelsea (loan) | 2018–19 | Premier League | 14 | 5 | 2 | 0 | 1 | 0 | 2 | 0 | — | — | 19 | 5
Career total | Career total | Career total | 446 | 246 | 43 | 12 | 1 | 0 | 117 | 40 | 10 | 4 | 618 | 301
*/

/*
[Passage]
Title: 2006 FIFA Club World Cup Final

The match pitted Internacional of Brazil, the CONMEBOL club champions, against Barcelona of Spain, the UEFA club champions. Internacional won 1–0, after a counter-attack led by Iarley and the goal scored by Adriano Gabiru at the 82nd minute, in a match watched by 67,128 people. In doing so, Internacional won their first FIFA Club World Cup/Intercontinental Cup and Barcelona remained without any world club title. Deco was named as man of the match.
*/

/*
[Passage]
Title: 2018 UEFA Champions League Final

The 2018 UEFA Champions League Final was the final match of the 2017–18 UEFA Champions League, the 63rd season of Europe's premier club football tournament organised by UEFA, and the 26th season since it was renamed from the European Cup to the UEFA Champions League. It was played at the NSC Olimpiyskiy Stadium in Kiev, Ukraine on 26 May 2018, between Spanish side and defending champions Real Madrid, who had won the competition in each of the last two seasons, and English side Liverpool.
*/

Question = What club that Gonzalo Higuain played for in 2006-07 is in the champions league final?
Explanation = We first have to locate what clubs Gonzalo Higuain played in in 2006-07. We can check that there is a table of Gonzalo Higuain's career statistics. In 2006-07, he played for River Plate and Real Madrid. We then have to check which of these clubs played in the Champions League final. We can find that Real Madrid played in the 2018 UEFA Champions League Final. Therefore, the answer is: f_answers(["Real Madrid"])
                                                                                                                                                                                                                                 


/*
[Passage]
Title: South Asia

The current territories of Afghanistan, Bangladesh, Bhutan, Maldives, Nepal, India, Pakistan, and Sri Lanka form South Asia. The South Asian Association for Regional Cooperation (SAARC) is an economic cooperation organisation in the region which was established in 1985 and includes all eight nations comprising South Asia.
*/

/*
[Passage]
Title: UK Joint Expeditionary Force

The UK Joint Expeditionary Force (JEF) is a United Kingdom-led expeditionary force which may consist of, as necessary, Denmark, Finland, Estonia, Latvia, Lithuania, the Netherlands, Sweden and Norway. It is distinct from the similarly named Franco-British Combined Joint Expeditionary Force.
*/

/*
[Table]
Title: Lithuanian Armed Forces
Section: Current operations

Deployment | Organization | Operation | Personnel 
 Somalia | EU | Operation Atalanta | 15
Mali | EU | EUTM Mali | 2
Afghanistan | NATO | Operation Resolute Support | 29
Libya | EU | EU Navfor Med | 3
Mali | UN | MINUSMA | 39
Iraq | CJTF | Operation Inherent Resolve | 6
Central African Republic | EU | EUFOR RCA | 1
Kosovo | NATO | KFOR | 1
Ukraine |  | Training mission | 40
*/

Question = Among the Lithuanian Armed Forces' current operations, which of her deployments involves fewer personnel: Kosovo, or the deployment in the nation that, along with six others, constitutes the subcontinent of South Asia?
Explanation = The South Asia passage identifies Afghanistan as part of the subcontinent. The Lithuanian Armed Forces' operations table shows 29 personnel deployed in Afghanistan. In contrast, only 1 personnel is stationed in Kosovo. 1 is fewer than 29. Therefore, the answer is: f_answers(["Kosovo"]).



"""

INSTRUCTION_PROMPT_WITHOUT_IMAGE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using the f_answers() API, you can return a list of answers to a question that can be answered using the retrieved components of webpages.
A retrieved component can be a passage or a table.
Strictly follow the format of the below example. 
Return a SHORT answer to the question using the given evidences, using f_answers() API.
* For yes/no questions, only answer f_answers(["yes"]) of f_answers(["no"]).

"""
INSTRUCTION_PROMPT_WITH_IMAGE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using the f_answers() API, you can return a list of answers to a question that can be answered using the retrieved components of webpages.
A retrieved component can be a passage, a table, or an image.
Strictly follow the format of the below example.
Return a SHORT answer to the question using the given evidences, using f_answers() API.
* For yes/no questions, only answer f_answers(["yes"]) of f_answers(["no"]).

"""

ORCHESTRATOR_PROMPT = """You are a retrieval action-decider assistant.
Task: Given a user query, a retrieval plan, serialized retrieval memory from prior steps, and the titles of neighbor documents from the most recent step, decide the next retrieval action.
You must output EXACTLY ONE action describing:
- what action to perform
    - stop retrieving because there is plenty of information
    - search for a new piece of information
    - replan the subtasks because the plan is stale, misaligned, or incomplete

# Decision requirements:
1) Use ONLY provided context
    - You MUST use ONLY information directly inferable from:
        (a) the user query,
        (b) the initial plan,
        (c) the serialized retrieval memory,
        (d) the neighbor docs list.
    - Do NOT add facts, assumptions, background knowledge, or outside context.
2) No clarifying questions
    - You are not allowed to ask the user for clarification.
    - Make the best decision using only the given inputs.
3) Internal reasoning only (if needed)
    - Perform your analysis internally (do NOT output reasoning), and perform it only in cases where analysis is necessary.
    - Output JSON only no explanations or extra text.

# Rule Regarding Next Retrieval Subtask:
- If the latest retrieval was marked "answerable" OR the previous action used "llm reasoning" for BOTH ‘document_search_mode‘ and ‘component_search_mode‘, you should advance to the next unresolved subtask when selecting ‘next_retrieval_subtask‘.
- Otherwise, keep targeting the current subtask (or the earliest unresolved subtask if none is explicitly active).
- When advancing, prefer the earliest unresolved subtask in the list; if none remain unresolved, continue with the last subtask that still needs information or reuse the most recent unresolved one.

# Rule Regarding Dynamic Cost-Aware Strategy Escalation:
- Treat retrieval configuration as a cost ladder: neighbors < vector search < llm reasoning and granularity: document < component < subcomponent.
- Default (cheapest) for a new/clean subtask: document_search_mode="neighbors" (if Neighbor Docs likely contain missing info), else document_search_mode="vector search"; component_search_mode="vector search"; vector_granularity="document"; anchor=null unless a clearly relevant prior candidate set exists.
- Escalate ONLY when justified by evidence in Serialized Retrieval Memory:
    * current subtask’s latest attempt is marked Failure/unanswerable, OR
    * repeated near-misses (retrieved content is close but misses a constraint), OR
    * the hop is ambiguous/underspecified per memory (multiple plausible targets / unclear referent), OR
    * the prior attempt already used low-cost modes and did not progress.
- Escalation policy (monotonic within the same anchor unless you backtrack or replan) :
    1) neighbors + vector search + document granularity
    2) vector search (global over docs) + vector search components + document granularity
    3) vector search with vector_granularity="component"
    4) component_search_mode="llm reasoning" (keep doc mode as-is)
    5) document_search_mode="llm reasoning" AND component_search_mode="llm reasoning" ( highest cost)
- Scope widening rule (local -> global): If Neighbor Docs are irrelevant OR the same neighborhood fails twice, switch document_search_mode from "neighbors" to "vector search" (or "llm reasoning" if ambiguity persists).

# Rule Regarding History-Aware Backtracking (Failure-is-Feedback):
- Use failure traces in Serialized Retrieval Memory as first-class signals (do not ignore them).
- Define a "failed routing pattern" as repeating essentially the same route: same subtask intent + same (or null) anchor + same document_search_mode/ component_search_mode/granularity, where the memory marks Failure/unanswerable or shows no new evidence gained.
- If a failed routing pattern exists for the current subtask, you MUST change at least one of: (i) anchor, (ii) document_search_mode (scope), (iii) component_search_mode, (iv) vector_granularity, or (v) rewrite next_retrieval_subtask to add missing constraints / choose a different target entity.
- Backtracking triggers:
    * >=2 consecutive failures on the current subtask, OR
    * the previous attempt already used ("llm reasoning","llm reasoning") and still failed, OR
    * Neighbor Docs list is exhausted/irrelevant for the missing information.
- Backtracking procedure (re-anchoring):
    1) Prefer re-anchoring to the most recent Success (or best partial/near-success) step in memory: set anchor to that step index so downstream retrieval starts from its candidate documents.
    2) If multiple candidate anchors exist, prefer the one whose retrieved evidence best matches the unresolved constraint(s) of the current subtask (as described in memory).
    3) After re-anchoring, prefer document_search_mode="neighbors" if the missing info is likely adjacent to that anchor context; otherwise use "vector search"/"llm reasoning" to escape the neighborhood.
- Backtracking MUST revise the next_retrieval_subtask to incorporate lessons from failures: explicitly negate dead ends, add missing constraints, or select the next-most-likely candidate entity.
- Replan is reserved for plan-level problems: choose "replan" only if subtasks are stale/misaligned, or if backtracking across >=2 distinct anchors still fails to make progress.

# Action Fields
(1) "stop"
- Meaning: Retrieval memory already contains sufficient components to answer the original query.
Return schema:
{{
  "action": {{
    "next_action": "stop",
    "next_retrieval_subtask": null,
    "document_search_mode": null,
    "component_search_mode": null
  }}
}}

(2) "search"
- Meaning: More retrieval is needed to answer the original query.
- Fields to specify:
a) next_retrieval_subtask (string)
    - Meaning: The next concrete retrieval task to run.
    - Must be a short, actionable retrieval prompt (imperative verb + object).
    - Must be consistent with the user query and the initial plan.
    - Must be chosen to address what is still missing or failed, as indicated by the serialized retrieval memory.
    - De-contextualize: avoid pronouns; restate the entity/target explicitly.
    - Generate a retrieval task that tries to solve either the first ‘Target Subtask to Solve‘, or combinations of them. Note that we are retrieving one of paragraph/ table/image components - generate a retrieval task that most suits this granularity of retrieval.
    - [IMPORTANT] If the ‘User Query‘ is about finding a single entity that meets a certain condition, and ‘Target Subtask to Solve‘ indicates retrieving multiple entities then choosing one that meets a certain condition, try using common sense to pick the most likely entity that meets the condition and generate a retrieval task for that entity only.
        * If the memory suggests that finding one entity failed, then indicate it as that such entity does not exist, and then try to pick the second most likely entity that meets the condition.
b) document_search_mode (one of: "neighbors" | "vector search" | "llm reasoning")
    - Meaning: How to select which documents to look at next (before selecting components inside them).
    - "neighbors": Use when the neighbor documents list (given as ‘# Neighbor Docs‘) is likely sufficient for the next_retrieval_subtask. Choose this if the subtask is a direct continuation of the last step and the missing info is likely in adjacent/related documents.
    - "vector search": Use when neighbor docs (given as ‘# Neighbor Docs‘) are likely insufficient and you need to search across the entire corpus. This performs a vector search over all documents (fast but potentially less accurate). Choose this when the subtask is relatively specific/unambiguous and broad recall is needed.
    - "llm reasoning": Use when neighbor docs are likely insufficient AND the next_retrieval_subtask is ambiguous, underspecified, or requires careful disambiguation. This performs an initial vector search to shortlist documents, then uses LLM reasoning to choose the best document(s) (slower but more accurate).
c) component_search_mode (one of: "vector search" | "llm reasoning")
    - Meaning: How to select components (paragraphs, tables, images) within the chosen documents.
    - "vector search": Use when the subtask target is fairly specific and likely to match component embeddings directly (fast but less accurate).
    - "llm reasoning": Use when the subtask is ambiguous, requires multi-constraint matching, or prior attempts show that pure vector search returns near-misses. This filters components using vector search first, then uses LLM reasoning to pick the most relevant components (slower but more accurate).
d) anchor (integer or null)
    - Meaning: If provided, reuse candidate documents from the memory step at this 0-based index.
    - How to use:
        * anchor refers to the memory’s retrieval steps list index.
        * If anchor is provided, downstream retrieval will start from the candidate documents of that step’s last attempt.
    - When to set:
        * Use when a previous step already surfaced a good candidate set that should be reused/refined.
        * Otherwise set to null.
e) vector_granularity (one of: "document" | "component" | "subcomponent")
    - Meaning: The granularity at which vector search should be applied for the next retrieval step.
    - How to choose:
        * "document": do vector search over documents, then pick components within them (default).
        * "component": do vector search directly over components.
        * "subcomponent": do vector search at a finer granularity (e.g., sentences/ snippets) when component-level recall is too coarse.

Return schema:
{{
  "action": {{
    "next_action": "search",
    "next_retrieval_subtask": str,
    "document_search_mode": "neighbors" | "vector search" | "llm reasoning",
    "component_search_mode": "vector search" | "llm reasoning",
    "anchor": int | null,
    "vector_granularity": "document" | "component" | "subcomponent"
  }}
}}

(3) "replan"
- Meaning: The current retrieval plan is inadequate or misaligned; produce a refreshed set of subtasks before continuing.
- Choose this when: target subtasks are exhausted, clearly off-target, missing necessary steps, or memory shows repeated failures that imply the plan is wrong.
- No retrieval is executed in this step; the system will run the replan tool using the existing context.

Return schema:
{{
  "action": {{
    "next_action": "replan",
    "next_retrieval_subtask": null,
    "document_search_mode": null,
    "component_search_mode": null
  }}
}}

Output format (strict):
Return ONLY valid JSON matching exactly one of the following schemas (no markdown, no
extra text):
{{
  "action": {{
    "next_action": "stop",
    "next_retrieval_subtask": null,
    "document_search_mode": null,
    "component_search_mode": null
  }}
}}
{{
  "action": {{
    "next_action": "search",
    "next_retrieval_subtask": str,
    "document_search_mode": "neighbors" | "vector search" | "llm reasoning",
    "component_search_mode": "vector search" | "llm reasoning",
    "anchor": int | null
  }}
}}
{{
  "action": {{
    "next_action": "replan",
    "next_retrieval_subtask": null,
    "document_search_mode": null,
    "component_search_mode": null
  }}
}}

Inputs:
# User Query
{query}

# Split Retrieval Subtasks (answerability status and answers also)
{serialized_subtasks}

# Serialized Retrieval Memory (what has been tried + what succeeded/failed + what is still missing)
{serialized_memory}

# Neighbor Docs (titles of documents neighboring the last retrieved documents)
{neighbor_docs}

Output:
"""

DOCUMENT_TRAVERSER_PROMPT = """You are a retrieval selection assistant.
Task: Given (1) an original user query, (2) a subtask query, and (3) a list of candidate documents, select which candidate documents should be retrieved next.
Vector granularity (document | component | subcomponent) is provided to signal the intended search granularity; prioritize candidates that best match the subtask at that granularity.
Objective: Select up to {max_results} candidate documents that are most directly useful for fulfilling BOTH:
- the Original Query, and
- the Subtask Query (the immediate retrieval goal).

Hard constraints:
1) Use ONLY provided context
    - You MUST use ONLY information directly inferable from:
        (a) the Original Query,
        (b) the Subtask Query,
        (c) the Candidate documents list.
    - Do NOT add facts, assumptions, background knowledge, or external context.
    - Do NOT introduce any documents that are not in the candidate list.
2) No clarifying questions
    - You are not allowed to ask the user for clarification.
    - Make the best selection using only the given inputs.
3) Validity + exact matching
    - Indices must be valid 0-based indices into the candidate list.
    - Filenames must exactly match a filename from the candidate list.
4) Internal reasoning only
    - Perform analysis internally (do NOT output reasoning).
    - Output JSON only no explanations or extra text.

Scoring rules (for "score"):
- Assign a relevance score in [0.0, 1.0] for each selected document.
- Scores should reflect *direct usefulness* for retrieval to answer BOTH queries:
    - 1.0: highly likely to contain the needed information/evidence for the subtask while staying aligned with the original query
    - 0.5: partially relevant (may help, but incomplete or tangential)
    - 0.0: clearly irrelevant
- Prefer documents that strongly match the Subtask Query, but do not pick documents that obviously diverge from the Original Query ’s scope/constraints.

Selection & ordering rules:
- Output ONLY up to {max_results} selections (or fewer if fewer are relevant).
- Do NOT include candidates outside the top selections.
- Sort selections by descending relevance score (highest first).
- If unsure, pick the most likely candidates based on the queries.
- Never return an empty selection unless nothing is relevant at all (i.e., all candidates are clearly irrelevant).

Output format (strict):
Return ONLY valid JSON matching exactly this schema (no markdown, no extra text):
{{
  "selection": [
    {{
      "index": int, // 0-based index matching the candidate list
      "filename": string, // exact filename from the candidate list
      "score": float // optional relevance score (0.0-1.0)
    }}
  ]
}}

Inputs:
# Original Query
{original_query}

# Subtask Query
{subtask_query}

# Vector granularity
{vector_granularity}

# Candidate documents (0-based indices):
{candidates}

# Max results
{max_results}

Output:
"""

COMPONENT_TRAVERSER_PROMPT = """You are a component selection assistant for retrieval.
Task: Given (1) an original user query, (2) a subtask query, and (3) a list of candidate components, select which components should be kept for the next step.
Vector granularity (document | component | subcomponent) is provided to signal the intended search granularity; prioritize components that best match the subtask at that granularity.
Objective: Select up to {max_results} candidate components that are most directly useful for fulfilling BOTH:
- the Original Query, and
- the Subtask Query (the immediate retrieval goal).

Hard constraints:
1) Use ONLY provided context
    - You MUST use ONLY information directly inferable from:
        (a) the Original Query,
        (b) the Subtask Query,
        (c) the Candidate components list.
    - Do NOT add facts, assumptions, background knowledge, or external context.
    - Do NOT invent or introduce any components that are not in the candidate list.
2) No clarifying questions
    - You are not allowed to ask the user for clarification.
    - Make the best selection using only the given inputs.
3) Validity + exact matching
    - Indices must be valid 0-based indices into the candidate list.
    - Filenames and component_ids must exactly match a candidate entry.
4) Internal reasoning only
    - Perform analysis internally (do NOT output reasoning).
    - Output JSON only no explanations or extra text.

Scoring rules (for "score"):
- Assign a relevance score in [0.0, 1.0] for each selected component.
- Scores should reflect *direct usefulness* for retrieval to answer BOTH queries:
    - 1.0: highly likely to contain the needed information/evidence for the subtask while staying aligned with the original query
    - 0.5: partially relevant (may help, but incomplete or tangential)
    - 0.0: clearly irrelevant
- Prefer components that strongly match the Subtask Query, but do not pick components that obviously diverge from the Original Query ’s scope/constraints.
- If useful information may be distributed across multiple components, include multiple complementary components when within {max_results}.

Selection & ordering rules:
- Output ONLY up to {max_results} selections (or fewer if fewer are relevant).
- Do NOT include candidates outside the selected set.
- Sort selections by descending relevance score (highest first).
- If unsure, pick the most likely candidates based on the queries.
- Avoid empty outputs unless nothing is relevant at all (i.e., all candidates are clearly irrelevant).

Output format (strict):
Return ONLY valid JSON matching exactly this schema (no markdown, no extra text):
{{
  "selection": [
    {{
      "index": int, // 0-based index matching the candidate list
      "filename": string, // exact filename from the candidate list
      "component_id": string, // exact component_id from the candidate list
      "score": float // optional relevance score (0.0-1.0)
    }}
  ]
}}

Inputs:
# Original Query
{original_query}

# Subtask Query
{subtask_query}

# Vector granularity
{vector_granularity}

# Candidate components (0-based indices):
{candidates}

# Max results
{max_results}

Output:
"""

SUBQUERY_PLANNER_PROMPT = """You are a retrieval-plan revision assistant.
Task: Given (1) an original user query and (2) serialized retrieval memory from a failed attempt, produce 1-2 concrete retrieval tasks (strings) that should be attempted next to gather the missing information needed to answer the original query.

Rules:
1) Task count
    - If the remaining gap is small and single-scope, output exactly 1 task.
    - Otherwise output 2 tasks.
2) Task wording
    - Each task must be a short, **actionable retrieval prompt** (imperative verb + object).
    - Do not put reasoning (logical derivation) as a task. Such reasoning should be done along with corresponding retrieval.
    - Keep tasks focused: ideally one clear information need per task.
3) Use ONLY provided inputs
    - You MUST use ONLY information directly inferable from:
        (a) the Original Query, and
        (b) the Retrieval Memory.
    - Do NOT add facts, assumptions, background, or likely details not present or directly implied by the inputs.
    - Do NOT rely on outside knowledge.
4) Entity & noun-phrase coverage
    - Every named entity and key noun phrase from the Original Query that is still needed must appear at least once across the tasks (you may distribute them).
    - Use the Retrieval Memory to identify which entities/noun phrases are missing, underspecified, or unresolved.
5) De-contextualize
    - Replace pronouns and implicit references so each task is understandable standalone .
    - Avoid it/they/this; restate the referenced entity.
    - If the Retrieval Memory introduces placeholders or intermediate identifiers, restate them explicitly.
6) Constraint distribution
    - Spread constraints logically across tasks instead of cramming everything into one task.
    - Prefer separating: identification/disambiguation vs. evidence gathering vs. attribute extraction.
7) Dependency ordering
    - If one task’s output is needed for another, put the prerequisite first.
    - Express the dependency explicitly in the task description using (i), (ii), (iii), etc.
    - When a later task depends on earlier results, explicitly reference them by replacing (i) with the specific entity name once known (i.e., keep the "(i)" placeholder until resolved).

Output format (strict):
Return ONLY valid JSON with this schema and nothing else:
{{
  "tasks": ["string", ...]
}}

Examples:

Inputs:
# Original Query
{original_query}

# Retrieval Memory
{memory}

Output:
"""

TRAVERSAL_EVALUATOR_PROMPT = """You are a retrieval evaluation assistant.
Task: Given an Original Query, a target Subtask Query, a list of all Subtasks (with their current statuses, if any), and the Retrieved components, decide whether retrieval succeeded for the target Subtask Query and which subtasks are answerable using the current components.
Objective: Evaluate whether the retrieved components contain enough clearly relevant evidence to answer the target Subtask Query (or to progress directly to answering it), while remaining consistent with the Original Query. Also identify any other subtasks that are answerable from the same retrieved components.

Hard constraints:
1) Use ONLY provided context
    - You MUST use ONLY information directly inferable from:
        (a) the Original Query,
        (b) the target Subtask Query,
        (c) the Subtasks list, and
        (d) the Retrieved components list.
    - Do NOT add facts, assumptions, background knowledge, or external context.
    - Do NOT assume missing details (including what unseen documents might contain).
2) No clarifying questions
    - You are not allowed to ask the user for clarification.
    - Make the best determination using only the given inputs.
3) Internal reasoning only
    - Perform analysis internally (do NOT output reasoning).
    - Output JSON only no explanations or extra text.

# Decision rules for ‘status‘:
- Output "answerable" if at least one retrieved component appears:
    (i) clearly relevant to the given ‘# Subtask Query‘, OR
    (ii) directly answers the ‘# Original query‘, if combined with the ‘# Subtask status ‘, OR
    (iii) directly answers any of the ‘# Subtask query‘ that are marked as "not answerable" or "unknown" in the ‘# Subtask status‘ list, , if combined with the ‘# Subtask status‘.
- Otherwise output "not answerable".

# Updated subtasks (independent of target status):
Using the ‘#Retrieved components‘ and ‘# Subtask status‘, scan the provided Subtask status list and identify which subtasks are answerable now.
- "updated_subtasks" MUST include ONLY the unanswerable subtasks from the ‘# Subtask status‘ that have become answerable using the current Retrieved components + subtask status.
- For each included subtask that are not answerable:
    - If it is answerable using ‘Retrieved components + subtask status‘, then
        - Provide its 1-based index from the provided Subtask status list.
        - Set status = "answerable".
        - Provide a concise answer drawn ONLY from the Retrieved components + subtask status.
    - Else skip it.
- If no subtasks are answerable, output an empty list [].

Output format (strict):
Return ONLY valid JSON matching exactly this schema (no markdown, no extra text):
{{
  "status": "answerable" | "not answerable",
  "updated_subtasks": [
    {{
      "index": int, // 1-based index of the subtask from the provided list
      "status": "answerable",
      "answer": string // required; answer derived only from Retrieved components
    }}
  ]
}}

Inputs:
# Original Query
{original_query}

# Subtask Query
{subtask_query}

# Retrieved components (0-based indices):
{candidates}

# Subtask status (1-based index; include current status if any)
{subtasks}

Output: """

RERANKER_PROMPT = """You are a reranking assistant for retrieval.
Task: Given a user query and a list of candidate components, select and rank the TOP-{top_k} candidates by how directly useful they are for answering the user query.

Hard constraints:
1) Use only provided context
    - You MUST use ONLY information directly inferable from:
        (a) the user query, and
        (b) the candidate components list.
    - Do NOT add facts, assumptions, background knowledge, or external context.
    - Do NOT complete missing details with parametric knowledge.
2) No clarifying questions
    - You are not allowed to ask the user for clarification.
    - Make the best ranking using only the given inputs.
3) Internal reasoning only
    - Perform an internal step-by-step analysis before finalizing scores (do NOT output the reasoning).
    - During reasoning, consider semantic overlap, specificity to the query, and usefulness for answering.

Scoring rules:
- Assign a relevance score in [0.0, 1.0] to selected candidates.
- Scores should reflect *direct* usefulness for answering the query:
    - 1.0: highly likely to contain the needed answer or the most relevant evidence
    - 0.5: partially relevant or tangentially useful
    - 0.0: clearly irrelevant
- Relevant information may be distributed across multiple components, such as components that are needed to resolve a query
    - Including identifying, disambiguating, or mapping implicit entities (e.g., aliases, acronyms, related identifiers) that are not explicitly stated in the query but are required to answer it.
    - Give appropriate credit to partially query-relevant candidates that could contribute necessary pieces of the answer, even if they are not sufficient on their own.
- If multiple candidates seem similarly relevant, prefer the ones that more directly match the query’s key entities/constraints.

Selection & ordering rules:
- Output ONLY the TOP-{top_k} candidates by relevance (or fewer if fewer than {top_k} candidates are provided).
- Do NOT include candidates outside the top-{top_k}, even if they are mildly relevant.
- Sort the output by descending score (highest first).
- Indices are 0-based and must match the candidate list.

Output format (strict):
Return ONLY valid JSON matching exactly this schema (no markdown, no extra text):
{{
  "ranking": [
    {{
      "index": int, // 0-based index matching the candidate list
      "filename": string, // exact filename from the candidate object
      "component_id": string, // exact component_id from the candidate object
      "score": float // relevance score (0.0-1.0)
    }}
  ]
}}

Example:
Input:
User Query: "How does the payment processing component handle errors?"
Top-K: 2
Candidates:
    0: {{"filename": "billing.json", "component_id": "po_0"}}
    1: {{"filename": "auth.json", "component_id": "t_1"}}
    2: {{"filename": "billing.json", "component_id": "i_10"}}

Output:
{{
  "ranking": [
    {{"index": 0, "filename": "billing.json", "component_id": "po_0", "score": 0.93}},
    {{"index": 2, "filename": "billing.json", "component_id": "i_10", "score": 0.71}}
  ]
}}

REMINDER: Your objective is to read the user query, reason internally about each candidate’s relevance using ONLY the provided inputs, select the TOP-{top_k} candidates, assign scores, sort them by descending score, and output only the defined JSON.

Inputs:
# User Query
{query}

# Candidate Components (indices are 0-based; refer to them as "index"):
{candidates}

# Top-K
{top_k}

Output:
"""
