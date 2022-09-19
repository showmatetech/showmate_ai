import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
BACK_URL = os.getenv("BACK_URL")


def model(df_songs, df_shows):
    print('Starting model...')

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

    # Under-sampling
    class_count_1, class_count_0 = df3_songs['score'].value_counts()
    class_0 = df3_songs[df3_songs['score'] == 0]
    class_1 = df3_songs[df3_songs['score'] == 1]

    target = class_count_0 + ((class_count_1 - class_count_0) / 2)

    target = int(target)

    class_1_under = class_1.sample(target)

    df_songs_under = pd.concat([class_1_under, class_0], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(df_songs_under.drop('score', axis=1),
                                                        df_songs_under.score,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=df_songs_under.score)

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

    print('Model OK!')

    return output
