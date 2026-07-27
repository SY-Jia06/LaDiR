import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vae"))

from data_vae import (  # noqa: E402
    extract_reasoning,
    infer_question,
    infer_responses,
    split_reasoning_blocks,
)


class DataVaeSchemaTests(unittest.TestCase):
    def test_input_output_schema(self):
        row = {"input": "What is 2+2?", "output": "Two plus two is four."}
        self.assertEqual(infer_question(row, None), "What is 2+2?")
        self.assertEqual(infer_responses(row, None), ["Two plus two is four."])

    def test_messages_schema(self):
        row = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "<think>Step one. Step two.</think> Final"},
            ]
        }
        self.assertEqual(infer_question(row, None), "Question")
        response = infer_responses(row, None)[0]
        self.assertEqual(extract_reasoning(response), "Step one. Step two.")

    def test_prompt_and_list_response_schema(self):
        row = {
            "prompt": "Solve the task",
            "small_model_responses": [],
            "strong_model_responses": [
                "<think>First observation. Second observation.</think> Answer"
            ],
        }
        self.assertEqual(infer_question(row, None), "Solve the task")
        responses = infer_responses(row, None, all_responses=True)
        self.assertEqual(len(responses), 1)
        self.assertIn("First observation", responses[0])

    def test_sentence_blocks(self):
        blocks = split_reasoning_blocks(
            "Let x be the unknown. Then 2x = 8. Therefore x = 4.",
            mode="sentence",
        )
        self.assertEqual(
            blocks,
            ["Let x be the unknown.", "Then 2x = 8.", "Therefore x = 4."],
        )

    def test_code_fence_is_one_block(self):
        text = "We can implement it directly.\n```python\nprint('ok')\n```\nThis finishes the proof."
        blocks = split_reasoning_blocks(text)
        self.assertIn("```python\nprint('ok')\n```", blocks)


if __name__ == "__main__":
    unittest.main()
