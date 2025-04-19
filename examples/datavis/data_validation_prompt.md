# Data Validation Prompt for LLM

This is the partial prompt used to validate if the scraped data matches the user's intent:

---

User intent: "<USER_QUERY>"
Here are the columns and first 5 rows of the data I found:

Columns: <COLUMNS>
Sample:
<DATA_SAMPLE>

Does this data match the user's request? Answer "yes" or "no" and explain briefly.

---

The LLM is expected to answer "yes" or "no" and provide a brief explanation. 