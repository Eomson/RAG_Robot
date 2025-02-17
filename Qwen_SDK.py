from langchain import LLMChain
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from langchain_core.prompts import PromptTemplate
from typing import List, Optional
import dashscope

# 自定义一个 LLM 类，用于调用 Qwen 的 API
class QwenAPI(BaseLLM):
    api_key: str

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        responses = []
        dashscope.api_key = self.api_key
        for prompt in prompts:
            try:
                response = dashscope.Generation.call(
                    model="qwen-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.7
                )
                if response.status_code == 200:
                    generated_text = response.output.text
                    responses.append([Generation(text=generated_text)])
                else:
                    raise Exception(f"API request failed with status code {response.status_code}: {response.message}")
            except Exception as e:
                raise Exception(f"Error during API call: {e}")

        return LLMResult(generations=responses)

    @property
    def _llm_type(self) -> str:
        return "qwen-api"

# 初始化 Qwen API
api_key = "your_api_key"  # 替换为你的 API 密钥
qwen_llm = QwenAPI(api_key=api_key)

# 定义对话模板
prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template="You are a helpful assistant. {input_text}"
)

# 创建 LLMChain
chain = LLMChain(llm=qwen_llm, prompt=prompt_template)

# 进行对话
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("Goodbye!")
        break
    response = chain.run(input_text=user_input)
    print(f"AI: {response}")