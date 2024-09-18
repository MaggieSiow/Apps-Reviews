from google_play_scraper import Sort, reviews_all
trip_advisor_review=reviews_all('com.spotify.music',
                           sleep_milliseconds=0,
                           lang='en',
                           country='us',
                           sort=Sort.NEWEST,
                           filter_score_with=None)

print(len(trip_advisor_review))

import pandas as pd
df=pd.DataFrame.from_records(trip_advisor_review)
df.head()
df.to_csv("C:/Users/lovep/OneDrive/Desktop/spotify.csv")

