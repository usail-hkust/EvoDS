CONTEXT_SUMMARIZE_PROMPT = """Your task is to summarize and distill all useful information obtained from the entire interaction history above, in order to support long-horizon decision making, planning, and tool usage in subsequent steps.

IMPORTANT:
- Do NOT summarize, restate, or infer the original problem statement.
- The original problem will be provided separately in a later step.
- Focus exclusively on summarizing information derived from the interaction history (e.g., decisions, constraints, intermediate results, environment or tool-related details).
- You must perform the summarization directly using your internal reasoning. Do NOT call, invoke, or rely on any external tools or functions during this process.

**Objectives**
Extract high-value, task-relevant information from the interaction history.
Remove redundancy, verbosity, and irrelevant conversational content.
Preserve actionable knowledge, constraints, decisions, and intermediate results.
Produce a compact yet information-complete summary that can be used as the agent’s persistent context or state.

**What to Include**
Summarize information including, but not limited to:
Environment-related information.
Data-related information.
Important decisions, conclusions, or resolved questions.
Derived insights, intermediate results, or partial solutions.
Tool usage rules, formats, or protocols established during the interaction.

**What to Exclude**
Polite language, greetings, or conversational fillers.
Repeated or superseded information.
Failed attempts unless they reveal an important constraint or insight.
Speculative or uncertain content not supported by the interaction.

**Output Requirements (Strict)**
The output must be a numbered list, strictly following the format:

1. …
2. …
3. …
   …

Each item must be:
Self-contained.
Concise yet precise.
Focused on a single piece of useful information.

Do NOT add headings, subheadings, or bullet points.
Do NOT include explanations outside the numbered list.
Do NOT explicitly reference the original conversation (e.g., “the user said above”).

**Role Awareness**
You are not solving the task itself.
You are producing a compressed, structured memory to be consumed in subsequent steps.
Ensure the summary is faithful, loss-minimized, and operationally useful.

Begin the summary now."""