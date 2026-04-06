"""
src/llm_corrector.py
LLM post-correction stage for OCR output.
Supports: Gemini (recommended, free tier) or OpenAI GPT-4o-mini.

Role in pipeline: LATE-STAGE cleanup only.
The CRNN does the heavy lifting; LLM corrects residual errors
using knowledge of 17th century Spanish language patterns.
"""

import os


class LLMCorrector:

    SYSTEM_PROMPT = (
        'You are an expert transcriber of 17th century Spanish printed sources. '
        'You will receive OCR output that may contain character recognition errors. '
        'Fix only clear OCR mistakes (e.g. rn->m, 0->o, I->l) using your knowledge '
        'of early modern Spanish spelling and vocabulary. '
        'Do NOT modernize spelling. Do NOT change meaning. '
        'Return only the corrected text, nothing else.'
    )

    def __init__(self, provider='gemini'):
        self.provider = provider
        self._client = None

        if provider == 'gemini':
            self._init_gemini()
        elif provider == 'openai':
            self._init_openai()
        else:
            raise ValueError(f'Unknown provider: {provider}')

    def _init_gemini(self):
        try:
            import google.generativeai as genai
            api_key = os.environ.get('GEMINI_API_KEY', '')
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel('gemini-2.0-flash')
            print('✓ Gemini client initialized')
        except Exception as e:
            print(f'⚠ Gemini init failed: {e}')
            self._client = None

    def _init_openai(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
            print('✓ OpenAI client initialized')
        except Exception as e:
            print(f'⚠ OpenAI init failed: {e}')
            self._client = None

    def correct(self, text: str) -> str:
        if not text or not text.strip() or self._client is None:
            return text
        try:
            if self.provider == 'gemini':
                return self._correct_gemini(text)
            else:
                return self._correct_openai(text)
        except Exception as e:
            print(f'LLM correction failed: {e}')
            return text

    def _correct_gemini(self, text: str) -> str:
        prompt = f'{self.SYSTEM_PROMPT}\n\nOCR text to correct:\n{text}'
        response = self._client.generate_content(prompt)
        return response.text.strip()

    def _correct_openai(self, text: str) -> str:
        response = self._client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': text}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()


print('✓ llm_corrector.py saved')
