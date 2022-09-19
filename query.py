import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")


def query(user_id):
    # Connect to MongoDB
    conn = MongoClient(MONGO_URI)
    db = conn["showMatchDB"]

    print('DB connection OK!')
    print('Starting query...')

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

    # Expand the cursor and construct the DataFrame
    u_df = pd.DataFrame(list(user_cursor))

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
    print('Query OK!')

    # Expand the cursor and construct the DataFrame
    e_df = pd.DataFrame(list(events_cursor))

    return u_df, e_df
