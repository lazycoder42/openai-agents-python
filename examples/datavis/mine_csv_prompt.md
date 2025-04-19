# LLM CSV Mining Prompt Template

User intent: "{user_query}"
Here are the first {n_lines} lines of the CSV file:

{sample_lines}

The column names and their meanings may vary. Infer which columns are relevant to the user's request based on the sample.
If you need to select columns for a time range, look for columns that look like years or dates (e.g., '2010', '2011', ...).
If you need to select a country, stock, or other entity, look for the most relevant column in the sample.
Use boolean indexing for row filtering and .loc[:, ...] for column selection in pandas.
If you need to select the last N columns, use df.loc[:, df.columns[-N:]].
To select year columns, you can use: [col for col in df.columns if col.isdigit() or col.startswith('20') or col.startswith('19')].
Assume the CSV file path is '{csv_file}' and use it directly in your code.
When filtering out rows with NaN or unavailable data, only drop rows if all relevant columns are NaN, or use a threshold if appropriate. It is common for some years to be missing in real-world data, so do not drop rows too aggressively.
The code should output the result as a DataFrame named result. Only output the code, nothing else. 