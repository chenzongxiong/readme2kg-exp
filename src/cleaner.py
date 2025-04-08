import re


class Cleaner:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    def clean(self):
        # TODO: implementation of various cleaning strategies for different scenarios
        return self.remove_markdown_quotes_if_needed().raw_text

    def remove_markdown_quotes_if_needed(self):
        self.raw_text = re.sub(r'^```markdown\n|```$',
                               '', self.raw_text.strip(), flags=re.MULTILINE)
        return self
