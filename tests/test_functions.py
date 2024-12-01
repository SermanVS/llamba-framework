from llamba_framework.chatmodels.chat_model import AbstractChatModel
import unittest

class TestAnalyzeFunction(unittest.TestCase):
    def test_query(self):        
        class DummyChatModel(AbstractChatModel): 
            def query(self, prompt, timeout):
                return "Nice question!"
        chat_model = DummyChatModel()
        prompt = f'What is C-reactive protein? What does an increased level of C-reactive protein mean?'
        res = chat_model.query(prompt=prompt, timeout=60)
        self.assertEqual(str.strip(res), "Nice question!")

if __name__ == '__main__':
    unittest.main()