import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(verbose=True)

api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key= os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = 'gpt-4o'
api_version = '2024-02-15-preview' # this might change in the future

aAzureOpenclient =  AsyncAzureOpenAI(
        api_key=api_key,  
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{deployment_name}"
    )


async def get_content_by_mulit_model(picture_url:str)->str:
    logging.info(f"Getting content by muliti model of image")

    response = await aAzureOpenclient.chat.completions.create(
        model=deployment_name,
        seed=99,
        messages=[
            { "role": "system", "content": "You are a helpful assistant and you are a teacher. You can give description or some keywords about image for students to understand the knowledge." },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": "Describe this picture in Chinese. Directly output the valid and useful information." 
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": picture_url
                    }
                }
            ] } 
        ],
        max_tokens=500 
    )

    return response.choices[0].message.content




if __name__ == "__main__":
    # 示例调用
    contentByMulitModel = asyncio.run(get_content_by_mulit_model("https://img2.tapimg.com/moment/etag/lhZEbeJKeI5qOwQxlRSUTsZcYen0.png"))
    print("contentByMulitModel: {}",contentByMulitModel)