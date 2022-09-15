import pandas as pd
from pymongo import MongoClient
import requests
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
BACK_URL = os.getenv("BACK_URL")


async def start_ai(user_id):
    print(user_id)
    # Connect to MongoDB
    conn = MongoClient(MONGO_URI)
    db = conn["showMatchDB"]

    # Make a query to the specific DB and Collection
    user_cursor = db["users"].aggregate([
        {"$match": {"userId": user_id}},
        {"$unwind": "$artists"},
        {"$replaceRoot": {"newRoot": "$artists"}},
        {"$unwind": "$artists"},
        {
            "$project": {
                "_id": 0,
                "score": "$score",
                "artistId": "$artists"
            }
        },
        {
            "$lookup":
                {
                    "from": "artists",
                    "localField": "artistId",
                    "foreignField": "artistId",
                    "as": "artistInfo"
                }
        },
        {
            "$replaceRoot": {"newRoot": {"$mergeObjects": [{"$arrayElemAt": ["$artistInfo", 0]}, "$$ROOT"]}}
        },
        {"$project": {"artistInfo": 0}},
        {"$unwind": "$topTracks"},
        {
            "$lookup":
                {
                    "from": "tracks",
                    "localField": "topTracks",
                    "foreignField": "trackId",
                    "as": "trackInfo"
                }
        },
        {
            "$replaceRoot": {"newRoot": {"$mergeObjects": [{"$arrayElemAt": ["$trackInfo", 0]}, "$$ROOT"]}}
        },
        {
            "$project": {
                "_id": 0,
                "topTracks": 0,
                "relatedArtists": 0,
                "trackInfo": 0,
                "uri": 0,
                "images": 0,
                "href": 0,
                "externalUrls": 0
            }
        }
    ])

    # var = list(user_cursor)
    # print(var)
    # Expand the cursor and construct the DataFrame
    u_df = pd.DataFrame(list(user_cursor))

    # Use DataFrame: u_df
    # ...

    events_cursor = db["users"].aggregate([
        {"$match": {"userId": user_id}},
        {"$unwind": "$events"},
        {
            "$project": {
                "_id": 0,
                "eventId": "$events"
            }
        },
        {
            "$lookup":
                {
                    "from": "events",
                    "localField": "eventId",
                    "foreignField": "eventId",
                    "as": "eventInfo"
                }
        },
        {
            "$replaceRoot": {"newRoot": {"$mergeObjects": [{"$arrayElemAt": ["$eventInfo", 0]}, "$$ROOT"]}}
        },
        {"$project": {"eventInfo": 0}},
        {"$unwind": "$artists"},
        {
            "$project": {
                "_id": 0,
                "eventId": "$eventId",
                "artistId": "$artists"
            }
        },
        {
            "$lookup":
                {
                    "from": "artists",
                    "localField": "artistId",
                    "foreignField": "artistId",
                    "as": "artistInfo"
                }
        },
        {
            "$replaceRoot": {"newRoot": {"$mergeObjects": [{"$arrayElemAt": ["$artistInfo", 0]}, "$$ROOT"]}}
        },
        {"$project": {"artistInfo": 0}},
        {"$unwind": "$topTracks"},
        {
            "$lookup":
                {
                    "from": "tracks",
                    "localField": "topTracks",
                    "foreignField": "trackId",
                    "as": "trackInfo"
                }
        },
        {
            "$replaceRoot": {"newRoot": {"$mergeObjects": [{"$arrayElemAt": ["$trackInfo", 0]}, "$$ROOT"]}}
        },
        {
            "$project": {
                "_id": 0,
                "topTracks": 0,
                "relatedArtists": 0,
                "trackInfo": 0,
                "uri": 0,
                "images": 0,
                "href": 0,
                "externalUrls": 0
            }
        }
    ])

    # Expand the cursor and construct the DataFrame
    e_df = pd.DataFrame(list(events_cursor))

    # Use DataFrame: e_df
    # ...

    data = {'user_id': user_id,
            'events': [
                {'eventId': '38937129',
                 'artistId': '2xe8IXgCTpwHE3eA9hTs4n',
                 'trackId': '7DF2SSCuHoniuh5sPeMvkv'
                 },
                {'eventId': '39255307',
                 'artistId': '25uiPmTg16RbhZWAqwLBy5',
                 'trackId': '27mT3JdR3sRJyiMBFHdhB4'
                 }
            ]}

    response = requests.post(url=BACK_URL, json=data)
    print(response)
    # if response.status == 200:
    #   print("Sucessfully send user events")
    # else:
    #   print("ERROR sending user events!")

    return u_df, e_df


# if __name__ == '__main__':
#   user_id = "rafzgz"
#   start_ai(user_id)
