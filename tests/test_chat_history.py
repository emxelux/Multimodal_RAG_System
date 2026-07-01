import unittest

from llm.ask_llm import build_history_messages


class ChatHistoryTests(unittest.TestCase):
    def test_build_history_messages_converts_prior_turns(self):
        history = [
            {"role": "user", "content": "What is this document about?"},
            {"role": "assistant", "content": "It discusses AI."},
            {"role": "user", "content": "Can you summarize it?"},
        ]

        messages = build_history_messages(history)

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].type, "human")
        self.assertEqual(messages[1].type, "ai")
        self.assertEqual(messages[2].type, "human")
        self.assertEqual(messages[0].content, "What is this document about?")
        self.assertEqual(messages[1].content, "It discusses AI.")
        self.assertEqual(messages[2].content, "Can you summarize it?")


if __name__ == "__main__":
    unittest.main()
