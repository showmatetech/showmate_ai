import pandas as pd
from pymongo import MongoClient
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
import requests
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
BACK_URL = os.getenv("BACK_URL")


def ai(df_songs, df_shows):
    df_songs = df_songs.drop('songkickArtistId', axis=1)

    df_songs = df_songs.dropna()

    df_songs = df_songs.drop('trackId', axis=1)

    df2_songs = df_songs.copy()

    df2_songs['liststring'] = [','.join(map(str, l)) for l in df2_songs['genres']]

    df_dummy_genre = df2_songs['liststring'].str.get_dummies(sep=',')

    df_dummy_genre_clean = df_dummy_genre[df_dummy_genre.columns[df_dummy_genre.sum() > 1000]]

    df3_songs = df2_songs.join(df_dummy_genre_clean)

    df3_songs = df3_songs.drop('genres', axis=1)

    df3_songs = df3_songs.drop('artistId', axis=1)

    df3_songs = df3_songs.drop('name', axis=1)

    df3_songs = df3_songs.drop('popularity', axis=1)

    df3_songs = df3_songs.drop('liststring', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df3_songs.drop('score', axis=1),
                                                        df3_songs.score,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=df3_songs.score)

    def build_model(hp):
        model = keras.Sequential()
        # Sample different number of layers with hp.Int
        for i in range(hp.Int('num_layers', 1, 3)):
            # Sample different number of layers with hp.Int
            model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=128,
                                                step=32),
                                   activation=hp.Choice('layers_activation', ['relu', 'tanh'])))
        # Set dense leyer automatically to 'sigmoid'
        model.add(keras.layers.Dense(1, 'sigmoid'))

        # Sample different activation functions with hp.Choice
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        return model

    tuner = kt.RandomSearch(build_model,
                            objective='accuracy',
                            max_trials=20,
                            overwrite=True)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train, y_train, epochs=30, validation_split=0.2, batch_size=32, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    df_pred = df_shows.copy()

    eventId = df_shows.pop('eventId')

    names = df_shows.pop('name')

    df_shows = df_shows.drop('songkickArtistId', axis=1)

    df_shows = df_shows.drop('trackId', axis=1)

    df_shows = df_shows.drop('popularity', axis=1)

    df_shows = df_shows.drop('artistId', axis=1)

    df_shows['liststring'] = [','.join(map(str, l)) for l in df_shows['genres']]

    df_shows_genres_dummies = df_shows['liststring'].str.get_dummies(sep=',')

    df_shows_2 = df_shows.join(df_shows_genres_dummies)

    df_shows_2 = df_shows_2.drop('genres', axis=1)

    df_shows_2 = df_shows_2.drop('liststring', axis=1)

    df_map = df3_songs.iloc[0:0]

    df_map = df_map.drop('score', axis=1)

    column_list_map = list(df_map)

    df_shows_final = pd.concat([df_map, df_shows_2], join='inner', ignore_index=True, axis=0)

    df_shows_final = df_shows_final.reindex(df_shows_final.columns.union(column_list_map, sort=False), axis=1,
                                            fill_value=0)

    predictions = model.predict(df_shows_final)

    predictions_bool = predictions > 0.5

    predictions_bin = predictions_bool.astype(int)

    df_pred['bin'] = predictions_bin

    df_pred_bin = df_pred[['bin', 'trackId', 'artistId', 'eventId']]

    pred_group = df_pred_bin.groupby(['trackId', 'artistId', 'eventId'], as_index=False)['bin'].sum().reset_index()

    matches_model_df = pred_group.loc[pred_group['bin'] > 3]

    matches_model_df_right = matches_model_df[['trackId', 'artistId', 'eventId']]

    matches_model_df_clean = matches_model_df_right.groupby(['artistId', 'eventId']).nth(0).reset_index()

    df_nonm_matches = df_songs.copy()

    df_nonm_matches2 = df_nonm_matches[['name', 'score']]

    df_nonm_matches_right = df_nonm_matches2.loc[df_nonm_matches2['score'] == 1]

    list_right_names = df_nonm_matches_right['name'].tolist()

    list_right_names_unique = list(set(list_right_names))

    df_matches_non = df_pred[df_pred['name'].isin(list_right_names_unique)]

    df_matches_non_rest = df_matches_non[['trackId', 'artistId', 'eventId']]

    matches_non_model_df_clean = df_matches_non_rest.groupby(['artistId', 'eventId']).nth(0).reset_index()

    frames = [matches_model_df_clean, matches_non_model_df_clean]

    final_pred_complete = pd.concat(frames)

    final_pred_complete_unique = final_pred_complete.drop_duplicates()

    output = final_pred_complete_unique.to_dict('records')

    return output


async def start_ai(user_id):
    print(user_id)
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

    # var = list(user_cursor)
    # print(var)
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
    print('Starting AI...')

    # Expand the cursor and construct the DataFrame
    e_df = pd.DataFrame(list(events_cursor))

    events = ai(u_df, e_df)

    print('AI OK!')
    print(events)
    data = {'user_id': user_id,
            'events': events}

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
