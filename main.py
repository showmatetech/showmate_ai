from fastapi import FastAPI
from ai import start_ai
import asyncio

app = FastAPI()


@app.get("/")
async def check():
    return {"success": True}


# noinspection PyAsyncCall
@app.get("/user/{user_id}")
async def process_user(user_id):
    asyncio.ensure_future(start_ai(user_id))
    print(user_id)
    return {"success": True}
