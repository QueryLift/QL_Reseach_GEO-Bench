class LLM_calls:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY が設定されていません")
        self.client = openai.OpenAI(api_key=api_key)
        self.async_client = openai.AsyncOpenAI(api_key=api_key)

        # レートリミットの設定
        self.rate_limit_interval = float(os.getenv("LLM_RATE_LIMIT_INTERVAL", "2.0"))
        self.rate_limit_lock = asyncio.Lock()
        self.last_call_time = 0.0

    @staticmethod
    def _extract_text(response) -> str:
        """Responses API は choices ではなく output_text プロパティを使う。"""
        text = (response.output_text or "").strip()
        if not text:
            raise RuntimeError("LLM からのレスポンスにテキストが含まれていません")
        return text

    def call_5_nano(self, prompt: str) -> str:
        response = self.client.responses.create(
            model="gpt-5-nano",
            input=prompt,
        )
        return self._extract_text(response)

    def call_5(self, prompt: str) -> str:
        response = self.client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            input=prompt,
        )
        return self._extract_text(response)

    async def acall_5_nano(self, prompt: str) -> str:
        async with self.rate_limit_lock:
            now = time.time()
            wait_time = self.rate_limit_interval - (now - self.last_call_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call_time = time.time()
        
        response = await self.async_client.responses.create(
            model="gpt-5-nano",
            input=prompt,
        )
        return self._extract_text(response)

    async def acall_5(self, prompt: str) -> str:
        async with self.rate_limit_lock:
            now = time.time()
            wait_time = self.rate_limit_interval - (now - self.last_call_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call_time = time.time()

        response = await self.async_client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            input=prompt,
        )
        return self._extract_text(response)

class GEO_bench:
    def __init__(self, llm: LLM_calls, target_contents: {"url": str, "content": str}):
        self.llm = llm
        self.target_contents = target_contents

    async def run(self, user_question):
        not_included_target_contents_answer, included_target_contents_answer = await self._run_with_target(user_question, included_target_contents=False), await self._run_with_target(user_question, included_target_contents=True)
        return not_included_target_contents_answer, included_target_contents_answer

    async def _run_with_target(self, user_question, included_target_contents=True):
        prompt = await self.generate_prompt(user_question, included_target_contents)
        answer = await self.llm.acall_5(prompt)
        return answer

    async def generate_prompt(self, user_question, included_target_contents=True):
        sources = await self._get_sources(user_question, included_target_contents)
        prompt = self._get_web_task_prompt(user_question, sources)
        return prompt

    async def _get_web_sources(self, prompts):
        sources = OpenAIのAPIを叩く,Sourceの一覧を取得
        return []

    async def _get_contents_from_web(self, url):
        return ""

    async def _get_sources(self, user_question, included_target_contents=False):
        source_links = await self._get_web_sources(user_question)
        sources = []
        for link in source_links:
            sources.append({
                url: source_links,
                content: await self._get_contents_from_web(link)
            })
        if included_target_contents:
            sources.append(self.target_contents) # target contentsを追加
        return sources


    def _get_web_task_prompt(self, user_question, sources):
        f"""
        1 Write an accurate and concise answer for the given user question,
        using _only_ the provided summarized web search results.
        The answer should be correct, high-quality, and written by
        an expert using an unbiased and journalistic tone. The user's language of choice such as English, Francais, Espamol,
        Deutsch, or should be used. The answer should be
        informative, interesting, and engaging. The answer's logic
        and reasoning should be rigorous and defensible. Every
        sentence in the answer should be _immediately followed_ by
        an in-line citation to the search result(s). The cited
        search result(s) should fully support _all_ the information
        in the sentence. Search results need to be cited using [
        index]. When citing several search results, use [1][2][3]format rather than [1, 2, 3]. You can use multiple search
        results to respond comprehensively while avoiding
        irrelevant search results.
        Question: {user_question}
        Search Results: {sources}
        """
        return ""