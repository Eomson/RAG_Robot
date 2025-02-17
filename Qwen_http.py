from langchain import LLMChain
from langchain.llms import BaseLLM
from langchain.schema import LLMResult,Generation
from langchain_core.prompts import PromptTemplate  # 使用正确的导入方式
from typing import List, Optional, Dict, Any
import requests

# 1. 自定义一个 LLM 类，用于调用 Qwen 的 API
class QwenAPI(BaseLLM):
    api_url: str  # 使用 Pydantic 定义字段
    api_key: str  # 使用 Pydantic 定义字段

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        responses = []
        for prompt in prompts:
            data = {
                "model": "qwen-turbo",  # 指定模型名称
                "input": {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                },
                "parameters": {
                    "max_tokens": 100,  # 控制生成文本的长度
                    "temperature": 0.7 # 控制生成文本的随机性
                }
            }
            response = requests.post(self.api_url, headers=headers, json=data)
            if response.status_code == 200:
                # print(response.json()) 
                # input()
                # {'output': {'finish_reason': 'stop', 'text': 'Hello! How can I assist you today?'}, 'usage': {'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 24, 'output_tokens': 9, 'input_tokens': 15}, 'request_id': '89674e31-bb42-9c90-854d-1338cea325a7'}

                generated_text = response.json()["output"]["text"]
                # 将字符串包装为 Generation 对象（关键修正！）
                responses.append([Generation(text=generated_text)])  # 注意这里是双重列表
            else:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        # 将结果包装为 LLMResult
        return LLMResult(generations=responses)

    @property
    def _llm_type(self) -> str:
        return "qwen-api"

# 2. 初始化 Qwen API
api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"  # 替换为实际的 Qwen API 地址
api_key = "your_api_key"  # 替换为你的 API 密钥
qwen_llm = QwenAPI(api_url=api_url, api_key=api_key)  # 初始化时传入参数

# 3. 定义对话模板
prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template="You are a helpful assistant. {input_text}"
)

# 4. 创建 LLMChain
chain = LLMChain(llm=qwen_llm, prompt=prompt_template)

# 5. 进行对话
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit","退出"]:
        print("Goodbye!")
        break
    response = chain.run(input_text=user_input)
    print(f"AI: {response}")
