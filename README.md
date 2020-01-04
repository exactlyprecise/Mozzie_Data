# Medium Docs link:

https://docs.google.com/document/d/1qwZhqqooVG5XMvkUdVPrp-sd_1HF5tTXXNeJ8xwqmuI/edit?usp=sharing

https://www.google.com/maps/d/viewer?mid=1MCdrIThPQYNI_wFNHNN3RKk9b3nKgLJe&ll=1.3164532575127086%2C103.80050000000006&z=12

For maps of stations:
http://www.weather.gov.sg/climate-historical-daily

Statistics of population:

https://www.singstat.gov.sg/find-data/search-by-theme/population/geographic-distribution/latest-data

Conventions for model training and documentation:
https://docs.google.com/document/d/14CumPA9qrzm4UgbWdu4Z207IkrD-B1KYKaRphHI6RTM/edit?usp=sharing

NOTES:
Ignore semakau data and sentosa island data.

Possible useful links:

https://github.com/ngbolin/DengAI

Chinese paper on year window dengue prediction:

https://www.biorxiv.org/content/biorxiv/early/2019/09/06/760702.full.pdf

Google groups:
https://groups.google.com/forum/#!forum/iot-datathon-30-aisavelives---dengue-outbreak-forecasting

Dengue NEA site (but no past data):
https://www.nea.gov.sg/dengue-zika/dengue/dengue-clusters

Documents Explanation:

datathon3-data/rainfall/_combined$PLACE$.csv: aggregating ranifall to temperature stations

datathon3-data/rainfall/_zeroes$PLACE.csv: just like combine place, but replacing unknown parts with 0.

datathon3-data/daily_tabulated_data.csv: combining the rainfall and temperature, daily

datathon3-data/weekly_tabulated_data.csv: averaging over temperature, and rafall_zeroes

weekly_labeled_data_deleted:
removed temperature stations and rainfall stations: sentosa, pulau ubin, semakau, jurong-island
khatib and tuas south: remove population (since they are zeros), but keep rainfall and temperature stations
