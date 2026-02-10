1. document summarization & embedding

2. structured memory
- history((action, anchor doc, comp, subquery, ) list)

3. traverser: want to find subquery q-relevant component
- hop
- llm-reasoning: 

4. subquery planner
- given full history, generate new subqueries

5. traversal evaluator
- judge whether the retrieved components can answer q_t
- check retrieved component can answer any remaining item of subquery list

6. reranker (end)
- rerank & select top-k

7. orchestrator
- make strategy tuple(hop scope, scoring granularity, reasoning)
- - select hop scope: local/global
- - select granularity: select granularity where the similarity score is calculated
- - llm reasoning: use llm based reasoning to accurately rerank components with q beyond vector scores.
