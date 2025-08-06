#!/bin/bash

# --- Script to wrap the 'claude' CLI tool with custom exit code logic ---

# Execute the command, capturing its combined standard output and standard error.
# The exit code is captured immediately after the command runs.
output=$(claude -p "/pre-commit $@" 2>&1)
exit_code=$?

# Always print the original output from the command so the user can see it.
echo "$output"

# --- Main Logic ---

# 1. If exit code is 1 AND the output is the specific API key error, exit with 0 (success override).
if [[ $exit_code -eq 1 && "$output" == "Invalid API key · Please run /login" ]]; then
  # This is the special case you want to treat as success.
  exit 0

# 2. If the exit code is non-zero (and wasn't the case above), propagate that error code.
elif [[ $exit_code -ne 0 ]]; then
  # For any other error, pass its exit code along.
  exit $exit_code

# 3. If the command was successful (exit code 0) AND the output contains "[PASS]", exit with 0.
elif [[ "$output" == *"[PASS]"* ]]; then
  # This is the expected success message.
  exit 0

# 4. Otherwise (command was successful but didn't have "[PASS]"), exit with 1 (failure).
else
  # The command ran without error, but the desired output was not found.
  exit 1
fi
