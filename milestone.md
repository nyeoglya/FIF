1. orchestrator => Current Working
- select hop scope: local/global
- select granularity: select granularity where the similarity score is calculated
- llm reasoning: use llm based reasoning to accurately rerank components with q beyond vector scores.

2. traverser: want to find subquery q-relevant component
- hop
- vector search
- llm-reasoning

3. subquery planner
- given full history, generate new subqueries

4. traversal evaluator
- judge whether the retrieved components can answer q_t
- check retrieved component can answer any remaining item of subquery list

5. reranker (end)
- rerank & select top-k

6. Evaluation
- Retrieval
- LLM Answer
