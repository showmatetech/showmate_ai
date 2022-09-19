from fastapi import FastAPI
from model import model
from query import query
import asyncio
from dotenv import load_dotenv
import requests
import os

BACK_URL = os.getenv("BACK_URL")

load_dotenv()
app = FastAPI()


async def process_user_data(user_id):
    print(user_id)
    u_df, e_df = query(user_id)
    events = model(u_df, e_df)
    print(events)
    data = {'user_id': user_id,
            'events': events}

    response = requests.post(url=BACK_URL, json=data)
    print(response)


@app.get("/")
async def check():
    asyncio.ensure_future(process_user_data('rafzgz'))
    print('rafzgz')
    return {"success": True}


@app.get("/user/{user_id}")
async def process_user(user_id):
    asyncio.ensure_future(process_user_data(user_id))
    print(user_id)
    return {"success": True}
