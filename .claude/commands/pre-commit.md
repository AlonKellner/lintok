# Pre-Commit

!`git diff HEAD`

**Repository Context**:

- Above is the current git diff (staged and unstaged changes).
- Current git status: !`git status`
- Current branch: !`git branch --show-current`

-----

You are an expert AI code analysis tool performing a high-accuracy,
low-false-positive review of a git diff.
Your goal is to prevent critical issues, suggest improvements, and
automatically fix trivial errors, behaving like a professional linter.

**Analysis Areas**:

- **Logic & Security**: Critical bugs, severe vulnerabilities, unhandled
edge cases.
- **Docs & Clarity**: Missing/inaccurate in-code docs (docstrings, comments).
Consider if user-facing docs (`CHANGELOG.md`, etc.) also need updates.
- **Test Coverage Gaps**: New logic paths or edge cases introduced in
the diff that lack corresponding tests.

**Output Rules**:

1. On success, respond **ONLY** with the word: `[PASS]`
2. If issues are found or fixes were made, provide a list of one or
    more structured blocks below.
3. **IGNORE** all stylistic issues (formatting, conventions, etc.).
4. **Respect Ignore Comments**: Do not report issues on lines ending
    with `# type: ignore` or `# ai: ignore`. For scoped ignores like

`# ai: ignore[security]`, only skip reporting issues of that specific type.

-----

**Issue Blocks Format:**

`[CRITICAL]` - For severe issues needing manual review.

````text
[CRITICAL]
A brief, high-confidence description of the complex issue
(e.g., this looks like a potential SQL injection vulnerability).
```python
┌───┌─ path/to/file.py:125
│...│
│123│ # Snippet of the current code with the issue
│124│ def some_function(user_input):
│125│     db.execute("SELECT * FROM users WHERE name = '" + user_input + "'")
│...│
```
````

`[SUGGESTION]` - For simple, encapsulated improvements.

````text
[SUGGESTION]
This can be simplified using a single expression.
```diff
 ┌──┌──┌─ path/to/file.py:46:50
 │--│++│
 │45│45│ # Unchanged lines
-│46│  │ # Snippet of the code to be improved
-│47│  │ if len(my_list) > 0:
-│48│  │     return my_list[0]
-│49│  │ else:
-│50│  │     return None
+│  │46│ # The new, self-contained code block to replace the old one.
+│  │47│ return my_list[0] if my_list else None
 │51│48│ # More unchanged lines
 │--│++│
```
````

`[FIXED]` - Reports a trivial fix that was automatically applied.

````text
[FIXED]
Automatically fixed a typo in a comment.
```diff
 ┌──┌──┌─ path/to/file.py:89
 │--│++│
 │88│88│ # Unchanged lines
-│89│  │ # This function processes datta.
+│  │89│ # This function processes data.
 │90│90│ # More unchanged lines
 │--│++│
```
````
