# Sample Questions for Manual Testing

Copy-paste these into the Streamlit UI (http://localhost:8501) to try each user
story. The curated companies are: **Microsoft**, **Toshiba**, **HCA Healthcare**,
**Insperity**, **1-800-FLOWERS.COM**, and **MGM Resorts International**.

These are a superset of `eval/questions.json` (which is for automated scoring) —
use this file for manually poking at the app and watching the step trace.

## Direct questions (single fact, one report)

Expect a grounded, cited answer.

- What was Microsoft's total revenue in fiscal year 2022?
- Approximately how many people did Microsoft employ as of June 30, 2022?
- Approximately how many employees did HCA Healthcare have as of December 31, 2022?
- How much did Insperity pay in dividends during 2022?
- What were Toshiba's net sales for the fiscal year, in yen?
- What were MGM Resorts International's consolidated net revenues in 2021?
- What were 1-800-FLOWERS.COM's total net revenues in fiscal 2022?
- By how much did Microsoft's research and development expenses increase in fiscal 2022?

## Compound questions (independent facts bundled together)

Expect one combined answer with the step trace showing the question split into
sub-questions.

- What was Microsoft's total revenue in fiscal 2022, and how many employees did HCA Healthcare have?
- How much did Insperity pay in dividends in 2022, and what were 1-800-FLOWERS.COM's total net revenues in fiscal 2022?
- What were Toshiba's net sales, and what were MGM Resorts International's net revenues in 2021?

## Comparison questions (uses the `compare_financial_values` tool)

Expect the step trace to show two retrieval sub-questions feeding a tool call,
and the final answer to state a computed result (which is greater, and by how
much), not a guess.

- Which company had more employees, Microsoft or HCA Healthcare, and by how much?
- Which had higher revenue, MGM Resorts International's 2021 net revenues or 1-800-FLOWERS.COM's fiscal 2022 total net revenues, and by how much?
- Which company paid more in dividends, Insperity in 2022 or Microsoft, and by how much?

## Edge cases (should fail gracefully, not fabricate)

- What is the capital of France? *(off-topic — should be declined, not answered)*
- What was Tesla's revenue in fiscal 2022? *(uncurated company — should say it doesn't have the information)*
- What was Microsoft's revenue in fiscal 2022, and what is Tesla's current market capitalization? *(compound, half-covered — should answer the Microsoft half and flag the rest as unknown)*
- Which had higher revenue, Amazon or Google? *(comparison with an uncurated company — should report it can't complete the comparison)*

## Things to watch in the step trace

- **classify_query**: did it correctly detect on-topic vs. off-topic, and simple vs. comparison?
- **route_subquestion**: did each sub-question get routed to `retrieval` (vs. `tool`)?
- **execute_retrieval**: which company/section was cited as the source?
- **execute_tool** (comparison only): what were the two parsed numeric inputs, and did the tool succeed or fail gracefully?
- **had_sufficient_info**: is it `False` exactly when the answer says it doesn't know?

Known real limitations (see README.md § 3 for the full write-up from the actual
evaluation run): retrieval sometimes surfaces the wrong company/section since
`nomic-embed-text` similarity scores cluster tightly in this corpus, and the 3B
model occasionally answers from outside the given context despite instructions
not to (e.g. the "capital of France" question). These are genuine findings from
running against a real local model, not scripted into this file.
