"""
Verify mathematical correctness of examples in cod_tasks_hard_similar.jsonl.
For each task, compare Output: values in the prompt against test_code assertions.
"""

import json
import re

DATA_PATH = r"c:\Users\cdici\OneDrive\Documenti\Personal Projects\ReasoningLab\data\cod_tasks_hard_similar.jsonl"


def extract_prompt_examples(prompt_text):
    """Extract (input_args, output_value) pairs from prompt Examples."""
    examples = []
    # Find all Example blocks
    example_blocks = re.split(r'Example \d+:', prompt_text)[1:]  # skip text before first example
    for block in example_blocks:
        # Extract Input line(s) and Output line
        input_match = re.search(r'Input:\s*(.+?)(?=Output:|$)', block, re.DOTALL)
        output_match = re.search(r'Output:\s*(.+?)(?=Explanation:|Example \d+:|Constraints:|$)', block, re.DOTALL)
        if input_match and output_match:
            input_str = input_match.group(1).strip()
            output_str = output_match.group(1).strip()
            # Take only first line of output (in case multi-line)
            output_str = output_str.split('\n')[0].strip()
            examples.append((input_str, output_str))
    return examples


def extract_assertions(test_code):
    """Extract expected values from assert statements."""
    assertions = []
    for line in test_code.split('\n'):
        line = line.strip()
        if line.startswith('assert '):
            # Find == and get expected value
            m = re.search(r'==\s*(.+)$', line)
            if m:
                expected = m.group(1).strip()
                assertions.append((line, expected))
    return assertions


def parse_value(s):
    """Try to parse a string as a Python literal."""
    s = s.strip()
    try:
        return eval(s)
    except Exception:
        return s  # return as string if can't eval


def check_task(task):
    """Check if prompt Output: values match test_code assertions.
    Returns list of discrepancy strings."""
    task_id = task.get('task_id', 'UNKNOWN')
    prompt = task.get('prompt', '')
    test_code = task.get('test_code', '')

    discrepancies = []

    prompt_examples = extract_prompt_examples(prompt)
    assertions = extract_assertions(test_code)

    if not prompt_examples:
        return discrepancies  # nothing to check

    # Parse prompt output values
    prompt_outputs = [parse_value(out) for (inp, out) in prompt_examples]
    # Parse assertion expected values
    assert_expected = [parse_value(exp) for (line, exp) in assertions]

    # Strategy: pair up prompt examples with assertions by order
    # Usually the first N assertions correspond to the N examples
    num_examples = len(prompt_outputs)
    num_assertions = len(assert_expected)

    for i, (prompt_out, (inp_str, out_str)) in enumerate(zip(prompt_outputs, prompt_examples)):
        if i < num_assertions:
            assert_val = assert_expected[i]
            assert_line = assertions[i][0]
            if prompt_out != assert_val:
                discrepancies.append(
                    f"  Example {i+1}: prompt says Output={repr(prompt_out)} "
                    f"but test_code expects {repr(assert_val)}\n"
                    f"    Prompt input: {inp_str}\n"
                    f"    Assert: {assert_line}"
                )
        else:
            # More examples than assertions - can't directly compare
            pass

    # Also check for self-contradictions in the explanation text
    # Look for Explanation blocks that mention a different value than Output
    explanation_blocks = re.findall(
        r'Output:\s*(.+?)\nExplanation:\s*(.+?)(?=\nExample \d+:|Constraints:|```|$)',
        prompt, re.DOTALL
    )
    for out_str, expl_str in explanation_blocks:
        out_str = out_str.strip().split('\n')[0].strip()
        expl_str = expl_str.strip()
        # Look for patterns like "The correct maximum is X" or "so the answer is X"
        corrections = re.findall(
            r'(?:correct (?:answer|maximum|result|output) is|answer is|result is|so (?:the )?(?:answer|result|output) is)\s*([0-9\[\]{}, ]+)',
            expl_str, re.IGNORECASE
        )
        for correction in corrections:
            correction = correction.strip().rstrip('.')
            try:
                corr_val = parse_value(correction)
                out_val = parse_value(out_str)
                if corr_val != out_val:
                    discrepancies.append(
                        f"  Self-contradiction: Output says {repr(out_val)} but explanation says 'correct answer is {repr(corr_val)}'\n"
                        f"    Explanation snippet: {expl_str[:200]}"
                    )
            except Exception:
                pass

    return discrepancies


def main():
    tasks = []
    with open(DATA_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    print(f"Loaded {len(tasks)} tasks.\n")

    all_issues = {}
    for task in tasks:
        task_id = task.get('task_id', 'UNKNOWN')
        issues = check_task(task)
        if issues:
            all_issues[task_id] = issues

    if not all_issues:
        print("No discrepancies found.")
    else:
        print(f"Found discrepancies in {len(all_issues)} task(s):\n")
        for task_id, issues in all_issues.items():
            print(f"Task {task_id}:")
            for issue in issues:
                print(issue)
            print()

    # Extra: dump all prompt outputs vs assertions for manual spot check
    print("\n--- Full comparison table (first 3 examples per task) ---")
    for task in tasks:
        task_id = task.get('task_id', 'UNKNOWN')
        prompt = task.get('prompt', '')
        test_code = task.get('test_code', '')
        prompt_examples = extract_prompt_examples(prompt)
        assertions = extract_assertions(test_code)
        prompt_outputs = [parse_value(out) for (inp, out) in prompt_examples]
        assert_expected = [parse_value(exp) for (line, exp) in assertions]
        # Only print if there's something to show
        rows = []
        for i, (po, ae) in enumerate(zip(prompt_outputs, assert_expected)):
            match = "OK" if po == ae else "MISMATCH"
            rows.append(f"  ex{i+1}: prompt={repr(po)} assert={repr(ae)} [{match}]")
        # Only print mismatches or first few
        mismatches = [r for r in rows if "MISMATCH" in r]
        if mismatches:
            print(f"Task {task_id}:")
            for r in mismatches:
                print(r)


if __name__ == '__main__':
    main()