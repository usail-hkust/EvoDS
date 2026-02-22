CONTINUE_PROMPT = """The problem to be solved is:
{task}

---
You already have access to a summarized record of the previous interaction history, distilled from earlier steps. The summary is as follows:
{summary}

Based on the problem definition and the summarized interaction history above, continue solving the problem.

When proceeding, you should:
Treat the provided summary as the authoritative representation of the prior context.
Leverage the extracted constraints, decisions, intermediate results, and insights.
Avoid revisiting or re-deriving information that has already been resolved unless strictly necessary.
Ensure consistency with previously established assumptions and tool usage rules.
Focus on making forward progress toward solving the problem in an efficient, coherent, and goal-directed manner.

Proceed with the next steps now."""