"""Fix inconsistencies in cod_tasks_hard_similar.jsonl.

Issues found:
1. LCB/48126: Output says 6, explanation/test say 7. Fix Output→7.
2. LCB/48144: Problem statement is confusing (disjoint subsequence vs overlapping subsets).
   Fix explanation to clarify the correct interpretation matching tests.
3. LCB/48158: "aaab" k=2 has 2 stable substrings ("aa" at pos 0-1 and 1-2), not 1.
   Fix output and test to 2.
4. LCB/48163: Output says 8, explanation/test say 10. Fix Output→10.
5. LCB/48175: Example 2 math is wrong. With no-3-consecutive constraint:
   Query1: nums=[5,4,5,-1], best={0,2}=10 (not 14).
   Query2: nums=[5,4,5,4], best={0,2,3}=14 (not 18).
   Total=24 (not 32). Fix explanation and test.
6. LCB/48182: Output says [1,2,1,1,1], should be [1,1,1,1,1]. Fix Output.
7. LCB/48194: Output says 1, explanation says 2. Fix Output→2.
8. LCB/48229: s="000111" k=1, best achievable max-run is 3, not 2.
   Fix: change to k=2 which does give answer 2. Update explanation and test.
9. LCB/48257: n=3,m=2 has 7 valid arrays (missing [2,2,2]). Fix output and test to 7.
"""
import json

INPUT = r"c:\Users\cdici\OneDrive\Documenti\Personal Projects\ReasoningLab\data\cod_tasks_hard_similar.jsonl"
OUTPUT = INPUT  # overwrite

with open(INPUT, encoding="utf-8") as f:
    lines = f.readlines()

tasks = [json.loads(line) for line in lines]

for t in tasks:
    tid = t["task_id"]

    # --- LCB/48126: Output 6→7 ---
    if tid == "LCB/48126":
        t["prompt"] = t["prompt"].replace(
            "Output: 6\nExplanation:\n"
            "The path 0 -> 1 -> 2 has values [2,2,3], so it is balanced and has total weight 7.\n"
            "But 0 -> 1 -> 3 has values [2,2,2], total weight 6.\n"
            "The correct maximum is 7.",
            "Output: 7\nExplanation:\n"
            "The path 0 -> 1 -> 2 has values [2,2,3], max - min = 1 <= 1, so it is balanced with total weight 4 + 3 = 7.\n"
            "The path 0 -> 1 -> 3 has values [2,2,2], total weight 4 + 2 = 6.\n"
            "The maximum is 7."
        )

    # --- LCB/48144: Fix explanations to match tests ---
    if tid == "LCB/48144":
        # Rewrite the problem statement for clarity and fix examples
        old_desc = (
            "You must choose a subsequence of indices of even size 2 * k for some k >= 1.\n"
            "Its score is defined as:\n\n"
            "(nums1[i_1] OR nums1[i_2] OR ... OR nums1[i_k]) XOR (nums2[i_{k+1}] OR nums2[i_{k+2}] OR ... OR nums2[i_{2k}])\n\n"
            "where i_1 < i_2 < ... < i_{2k} are the chosen indices.\n"
            "Return the maximum possible score."
        )
        new_desc = (
            "You must choose two non-empty subsets of indices A and B, each of size k for some k >= 1. "
            "The subsets may overlap.\n"
            "The score is defined as:\n\n"
            "(OR of nums1[i] for i in A) XOR (OR of nums2[j] for j in B)\n\n"
            "Return the maximum possible score."
        )
        t["prompt"] = t["prompt"].replace(old_desc, new_desc)

        # Fix example 1 explanation
        t["prompt"] = t["prompt"].replace(
            "Choose both indices. Score = 1 XOR 7 = 6.",
            "Choose A = {1}, B = {1}. Score = nums1[1] XOR nums2[1] = 2 XOR 4 = 6."
        )

        # Fix example 2 explanation
        t["prompt"] = t["prompt"].replace(
            "Choose indices [1,2]. Score = 4 XOR 7 = 3.\n"
            "Choosing all three is not allowed since the size must be even.\n"
            "The best even-sized subsequence gives 6.",
            "Choose A = {0}, B = {2}. Score = nums1[0] XOR nums2[2] = 1 XOR 7 = 6."
        )

    # --- LCB/48158: "aaab" k=2 → 2 stable substrings, not 1 ---
    if tid == "LCB/48158":
        t["prompt"] = t["prompt"].replace(
            'Input: s = "aaab", k = 2\nOutput: 1\nExplanation:\nOnly "aa" is stable.',
            'Input: s = "aaab", k = 2\nOutput: 2\nExplanation:\nThe two stable substrings are "aa" starting at index 0 and "aa" starting at index 1.'
        )
        t["test_code"] = t["test_code"].replace(
            "sol.countStableSubstrings('aaab', 2) == 1",
            "sol.countStableSubstrings('aaab', 2) == 2"
        )

    # --- LCB/48163: Output 8→10 ---
    if tid == "LCB/48163":
        t["prompt"] = t["prompt"].replace(
            "Output: 8\nExplanation:\n"
            "A maximum-sum smooth subsequence is [4,6] with sum 10, but |6 - 4| = 2 so it is valid.\n"
            "Therefore the answer is 10.",
            "Output: 10\nExplanation:\n"
            "The smooth subsequence [4,6] has sum 10 and |6 - 4| = 2 <= d, so it is valid.\n"
            "No smooth subsequence has a larger sum."
        )

    # --- LCB/48175: Fix example 2 math ---
    if tid == "LCB/48175":
        t["prompt"] = t["prompt"].replace(
            "Output: 18\nExplanation:\n"
            "After first update nums = [5,4,5,-1], maximum sum is 14.\n"
            "After second update nums = [5,4,5,4], maximum sum is 18 by choosing indices 0,1,3.\n"
            "Total is 32.",
            "Output: 24\nExplanation:\n"
            "After first update nums = [5,4,5,-1], maximum sum is 10 (choose indices 0 and 2).\n"
            "After second update nums = [5,4,5,4], maximum sum is 14 (choose indices 0, 2, and 3).\n"
            "Total is 10 + 14 = 24."
        )
        t["test_code"] = t["test_code"].replace(
            "sol.maximumSubsequenceSumNoThree([5, -1, 5, -1], [[1, 4], [3, 4]]) == 32",
            "sol.maximumSubsequenceSumNoThree([5, -1, 5, -1], [[1, 4], [3, 4]]) == 24"
        )

    # --- LCB/48182: Output [1,2,1,1,1]→[1,1,1,1,1] ---
    if tid == "LCB/48182":
        t["prompt"] = t["prompt"].replace(
            "Output: [1,2,1,1,1]\nExplanation:\n"
            "Subtree of 0 has values [1,2,1,2,3], only 3 appears exactly once.\n"
            "Subtree of 1 has values [2,2,3], only 3 appears exactly once.",
            "Output: [1,1,1,1,1]\nExplanation:\n"
            "Subtree of 0 has values [1,2,1,2,3], only 3 appears exactly once, so score = 1.\n"
            "Subtree of 1 has values [2,2,3], only 3 appears exactly once, so score = 1.\n"
            "Each leaf subtree contains a single value appearing once, so score = 1."
        )

    # --- LCB/48194: Output 1→2, fix explanation ---
    if tid == "LCB/48194":
        t["prompt"] = t["prompt"].replace(
            "Output: 1\nExplanation:\n"
            "Partition as [1] and [3,2]. Costs are 0 and 1 * 2 = 2? That gives 2.\n"
            "Better partition is [1,3] and [2], cost 2 * 2 + 0 = 4.\n"
            "Actually the optimal is [1],[3],[2] not allowed.\n"
            "So the answer is 2.",
            "Output: 2\nExplanation:\n"
            "Partition as [1] and [3,2]. Costs are 0 + (3-2)*2 = 2.\n"
            "Partition as [1,3] and [2]. Costs are (3-1)*2 + 0 = 4.\n"
            "The optimal partition gives a total cost of 2."
        )

    # --- LCB/48229: k=1→k=2 for example 1, fix explanation and test ---
    if tid == "LCB/48229":
        t["prompt"] = t["prompt"].replace(
            'Input: s = "000111", k = 1\nOutput: 2\nExplanation:\n'
            'Flip one character to obtain "001111" or "000110"? Those are not optimal.\n'
            'By flipping the third character we get "001111" with maximum run 4.\n'
            'Better is flipping one of the middle characters to obtain "001011" with maximum run 2.',
            'Input: s = "000111", k = 2\nOutput: 2\nExplanation:\n'
            'Flip positions 1 and 5 to obtain "010110", which has a maximum run of 2.\n'
            'No arrangement with 2 flips can achieve a maximum run of 1.'
        )
        t["test_code"] = t["test_code"].replace(
            "sol.minimizeLongestRun('000111', 1) == 2",
            "sol.minimizeLongestRun('000111', 2) == 2"
        )

    # --- LCB/48257: n=3, m=2 → 7 valid arrays, not 6 ---
    if tid == "LCB/48257":
        t["prompt"] = t["prompt"].replace(
            "Output: 6\nExplanation:\n"
            "The valid arrays are [1,1,1], [1,1,2], [2,1,1], [2,2,1], [2,1,2], [1,2,1].",
            "Output: 7\nExplanation:\n"
            "The valid arrays are [1,1,1], [1,1,2], [2,1,1], [2,2,1], [2,1,2], [1,2,1], and [2,2,2].\n"
            "The only invalid array is [1,2,2], since arr[1]=2 > arr[0]=1 but arr[1] is not > arr[2]=2."
        )
        t["test_code"] = t["test_code"].replace(
            "sol.countPeakValidArrays(3, 2) == 6",
            "sol.countPeakValidArrays(3, 2) == 7"
        )

with open(OUTPUT, "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

print("Done. Fixed 9 issues across 9 tasks.")
