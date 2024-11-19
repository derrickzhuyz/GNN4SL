Link-level graph datasets: for each dataset, the number of generated graphs equals to the number of databases

Inspection Example:

```json
=== Inspecting Link-level Graph: 0 ===

Graph Structure:
Data(x=[70, 3], edge_index=[2, 193], node_names=[70], node_types=[70], database_name='concert_singer')

Graph Statistics:
Number of nodes: 70
Number of edges: 193
Average degree: 5.51

Node Types Distribution:
  - table: 4
  - column: 21
  - question: 45

Node Information:
Node 0: stadium (table)
  - Embedding (first three values): [0.07861121 0.48702663 0.6872855 ]
Node 1: stadium_id (column)
  - Embedding (first three values): [0.33337724 0.79151    0.1838022 ]
Node 2: location (column)
  - Embedding (first three values): [0.5894001  0.5754917  0.70753515]
Node 3: name (column)
  - Embedding (first three values): [0.2785914  0.308952   0.08608443]
Node 4: capacity (column)
  - Embedding (first three values): [0.3890879  0.5893352  0.34179825]
Node 5: highest (column)
  - Embedding (first three values): [0.67118883 0.01473221 0.78720355]
Node 6: lowest (column)
  - Embedding (first three values): [0.26065654 0.5880603  0.6401594 ]
Node 7: average (column)
  - Embedding (first three values): [0.921932  0.2910665 0.8168265]
Node 8: singer (table)
  - Embedding (first three values): [0.54271084 0.64325535 0.11860269]
Node 9: singer_id (column)
  - Embedding (first three values): [0.1333736  0.09488335 0.8006504 ]
Node 10: name (column)
  - Embedding (first three values): [0.54761136 0.0276709  0.28426152]
Node 11: country (column)
  - Embedding (first three values): [0.58892316 0.5464688  0.3507285 ]
Node 12: song_name (column)
  - Embedding (first three values): [0.340062   0.96974266 0.55370545]
Node 13: song_release_year (column)
  - Embedding (first three values): [0.6664307 0.9498801 0.7546454]
Node 14: age (column)
  - Embedding (first three values): [0.31199104 0.24097204 0.07128235]
Node 15: is_male (column)
  - Embedding (first three values): [0.9242103  0.17472635 0.7158627 ]
Node 16: concert (table)
  - Embedding (first three values): [0.5070624  0.6325287  0.92787063]
Node 17: concert_id (column)
  - Embedding (first three values): [0.7032236  0.14594649 0.97615886]
Node 18: concert_name (column)
  - Embedding (first three values): [0.06982073 0.91768193 0.9711458 ]
Node 19: theme (column)
  - Embedding (first three values): [0.7929897  0.99647254 0.507424  ]
Node 20: stadium_id (column)
  - Embedding (first three values): [0.26661798 0.7336045  0.22538751]
Node 21: year (column)
  - Embedding (first three values): [0.24793766 0.51838195 0.20411338]
Node 22: singer_in_concert (table)
  - Embedding (first three values): [0.9673047  0.6608301  0.43312776]
Node 23: concert_id (column)
  - Embedding (first three values): [0.64874697 0.35169753 0.51186275]
Node 24: singer_id (column)
  - Embedding (first three values): [0.5102229  0.00351903 0.29235289]
Node 25: How many singers do we have? (question)
  - Embedding (first three values): [0.41906294 0.73421    0.78431165]
Node 26: What is the total number of singers? (question)
  - Embedding (first three values): [0.94019765 0.30458587 0.6267495 ]
Node 27: Show name, country, age for all singers ordered by age from the oldest to the youngest. (question)
  - Embedding (first three values): [0.08059821 0.09286082 0.28348577]
Node 28: What are the names, countries, and ages for every singer in descending order of age? (question)
  - Embedding (first three values): [0.19244201 0.45980692 0.65884656]
Node 29: What is the average, minimum, and maximum age of all singers from France? (question)
  - Embedding (first three values): [0.5329093  0.37944084 0.55542845]
Node 30: What is the average, minimum, and maximum age for all French singers? (question)
  - Embedding (first three values): [0.33466628 0.6476107  0.5430368 ]
Node 31: Show the name and the release year of the song by the youngest singer. (question)
  - Embedding (first three values): [0.6836775  0.9610442  0.31307578]
Node 32: What are the names and release years for all the songs of the youngest singer? (question)
  - Embedding (first three values): [0.33404148 0.8770349  0.3972239 ]
Node 33: What are all distinct countries where singers above age 20 are from? (question)
  - Embedding (first three values): [0.9476351  0.04973063 0.2874459 ]
Node 34: What are  the different countries with singers above age 20? (question)
  - Embedding (first three values): [0.6551963  0.6344287  0.61839604]
Node 35: Show all countries and the number of singers in each country. (question)
  - Embedding (first three values): [0.9211075  0.34980094 0.8077854 ]
Node 36: How many singers are from each country? (question)
  - Embedding (first three values): [0.5136689  0.8305701  0.24303575]
Node 37: List all song names by singers above the average age. (question)
  - Embedding (first three values): [0.7160355  0.21065804 0.55232763]
Node 38: What are all the song names by singers who are older than average? (question)
  - Embedding (first three values): [0.26901013 0.9316762  0.4369489 ]
Node 39: Show location and name for all stadiums with a capacity between 5000 and 10000. (question)
  - Embedding (first three values): [0.30911866 0.18625696 0.90873814]
Node 40: What are the locations and names of all stations with capacity between 5000 and 10000? (question)
  - Embedding (first three values): [0.5721764  0.6332634  0.08922567]
Node 41: What is the average and the maximum capacity of all stadiums? (question)
  - Embedding (first three values): [0.15892717 0.554179   0.72292954]
Node 42: What is the average and maximum capacities for all stations? (question)
  - Embedding (first three values): [0.8182627  0.35315022 0.0773718 ]
Node 43: What is the name and capacity for the stadium with highest average attendance? (question)
  - Embedding (first three values): [0.65654004 0.12551452 0.00514136]
Node 44: What is the name and capacity for the stadium with the highest average attendance? (question)
  - Embedding (first three values): [0.7795608  0.94222826 0.23410755]
Node 45: How many concerts are there in year 2014 or 2015? (question)
  - Embedding (first three values): [0.75016886 0.04327605 0.04619223]
Node 46: How many concerts occurred in 2014 or 2015? (question)
  - Embedding (first three values): [0.55152154 0.01699318 0.06630405]
Node 47: Show the stadium name and the number of concerts in each stadium. (question)
  - Embedding (first three values): [0.87748206 0.78190607 0.13430972]
Node 48: For each stadium, how many concerts play there? (question)
  - Embedding (first three values): [0.38194382 0.02985267 0.19957122]
Node 49: Show the stadium name and capacity with most number of concerts in year 2014 or after. (question)
  - Embedding (first three values): [0.8143166 0.2810933 0.5884035]
Node 50: What is the name and capacity of the stadium  with the most concerts after 2013? (question)
  - Embedding (first three values): [0.3078454  0.00401077 0.00263552]
Node 51: Which year has most number of concerts? (question)
  - Embedding (first three values): [0.58102155 0.0196259  0.03789872]
Node 52: What is the year that had the most concerts? (question)
  - Embedding (first three values): [0.68014085 0.21543269 0.6230705 ]
Node 53: Show the stadium names without any concert. (question)
  - Embedding (first three values): [0.40829882 0.73926723 0.27295077]
Node 54: What are the names of the stadiums without any concerts? (question)
  - Embedding (first three values): [0.8342494  0.66056716 0.14980012]
Node 55: Show countries where a singer above age 40 and a singer below 30 are from. (question)
  - Embedding (first three values): [0.83017707 0.29619458 0.40666896]
Node 56: Show names for all stadiums except for stadiums having a concert in year 2014. (question)
  - Embedding (first three values): [0.11557456 0.23057005 0.2675674 ]
Node 57: What are the names of all stadiums that did not have a concert in 2014? (question)
  - Embedding (first three values): [0.7601736  0.10197102 0.7277771 ]
Node 58: Show the name and theme for all concerts and the number of singers in each concert. (question)
  - Embedding (first three values): [0.53479207 0.98382086 0.54010713]
Node 59: What are the names, themes, and number of singers for each and every concert? (question)
  - Embedding (first three values): [0.9917563  0.14629468 0.16457072]
Node 60: List singer names and number of concerts for each singer. (question)
  - Embedding (first three values): [0.9779802 0.769369  0.700003 ]
Node 61: What are the names of the singers and number of concerts for each person? (question)
  - Embedding (first three values): [0.33472127 0.207337   0.13883954]
Node 62: List all singer names in concerts in year 2014. (question)
  - Embedding (first three values): [0.45548114 0.09270477 0.47265124]
Node 63: What are the names of the singers who performed in a concert in 2014? (question)
  - Embedding (first three values): [0.2929488 0.8997723 0.0702453]
Node 64: what is the name and nation of the singer who have a song having 'Hey' in its name? (question)
  - Embedding (first three values): [0.54567575 0.87482333 0.5327663 ]
Node 65: What is the name and country of origin of every singer who has a song with the word 'Hey' in its title? (question)
  - Embedding (first three values): [0.6942269  0.21847758 0.14121382]
Node 66: Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015. (question)
  - Embedding (first three values): [0.5744574  0.65361226 0.8958824 ]
Node 67: What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015? (question)
  - Embedding (first three values): [0.2264076  0.48513085 0.8295723 ]
Node 68: Find the number of concerts happened in the stadium with the highest capacity. (question)
  - Embedding (first three values): [0.05513292 0.4387618  0.6681038 ]
Node 69: What are the number of concerts that occurred in the stadium with the largest capacity? (question)
  - Embedding (first three values): [0.19300245 0.02672816 0.30643183]

Edge Information:
Edge: 0(table:stadium) <-> 1(column:stadium_id)
Edge: 0(table:stadium) <-> 2(column:location)
Edge: 0(table:stadium) <-> 3(column:name)
Edge: 0(table:stadium) <-> 4(column:capacity)
Edge: 0(table:stadium) <-> 5(column:highest)
Edge: 0(table:stadium) <-> 6(column:lowest)
Edge: 0(table:stadium) <-> 7(column:average)
Edge: 0(table:stadium) <-> 39(question:Show location and name for all stadiums with a capacity between 5000 and 10000.)
Edge: 0(table:stadium) <-> 40(question:What are the locations and names of all stations with capacity between 5000 and 10000?)
Edge: 0(table:stadium) <-> 41(question:What is the average and the maximum capacity of all stadiums?)
Edge: 0(table:stadium) <-> 42(question:What is the average and maximum capacities for all stations?)
Edge: 0(table:stadium) <-> 43(question:What is the name and capacity for the stadium with highest average attendance?)
Edge: 0(table:stadium) <-> 44(question:What is the name and capacity for the stadium with the highest average attendance?)
Edge: 0(table:stadium) <-> 47(question:Show the stadium name and the number of concerts in each stadium.)
Edge: 0(table:stadium) <-> 48(question:For each stadium, how many concerts play there?)
Edge: 0(table:stadium) <-> 49(question:Show the stadium name and capacity with most number of concerts in year 2014 or after.)
Edge: 0(table:stadium) <-> 50(question:What is the name and capacity of the stadium  with the most concerts after 2013?)
Edge: 0(table:stadium) <-> 53(question:Show the stadium names without any concert.)
Edge: 0(table:stadium) <-> 54(question:What are the names of the stadiums without any concerts?)
Edge: 0(table:stadium) <-> 56(question:Show names for all stadiums except for stadiums having a concert in year 2014.)
Edge: 0(table:stadium) <-> 57(question:What are the names of all stadiums that did not have a concert in 2014?)
Edge: 0(table:stadium) <-> 66(question:Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015.)
Edge: 0(table:stadium) <-> 67(question:What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?)
Edge: 0(table:stadium) <-> 68(question:Find the number of concerts happened in the stadium with the highest capacity.)
Edge: 0(table:stadium) <-> 69(question:What are the number of concerts that occurred in the stadium with the largest capacity?)
Edge: 1(column:stadium_id) <-> 20(column:stadium_id)
Edge: 2(column:location) <-> 39(question:Show location and name for all stadiums with a capacity between 5000 and 10000.)
Edge: 2(column:location) <-> 40(question:What are the locations and names of all stations with capacity between 5000 and 10000?)
Edge: 2(column:location) <-> 66(question:Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015.)
Edge: 2(column:location) <-> 67(question:What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?)
Edge: 4(column:capacity) <-> 39(question:Show location and name for all stadiums with a capacity between 5000 and 10000.)
Edge: 4(column:capacity) <-> 40(question:What are the locations and names of all stations with capacity between 5000 and 10000?)
Edge: 4(column:capacity) <-> 41(question:What is the average and the maximum capacity of all stadiums?)
Edge: 4(column:capacity) <-> 42(question:What is the average and maximum capacities for all stations?)
Edge: 4(column:capacity) <-> 43(question:What is the name and capacity for the stadium with highest average attendance?)
Edge: 4(column:capacity) <-> 44(question:What is the name and capacity for the stadium with the highest average attendance?)
Edge: 4(column:capacity) <-> 49(question:Show the stadium name and capacity with most number of concerts in year 2014 or after.)
Edge: 4(column:capacity) <-> 50(question:What is the name and capacity of the stadium  with the most concerts after 2013?)
Edge: 4(column:capacity) <-> 68(question:Find the number of concerts happened in the stadium with the highest capacity.)
Edge: 4(column:capacity) <-> 69(question:What are the number of concerts that occurred in the stadium with the largest capacity?)
Edge: 7(column:average) <-> 43(question:What is the name and capacity for the stadium with highest average attendance?)
Edge: 7(column:average) <-> 44(question:What is the name and capacity for the stadium with the highest average attendance?)
Edge: 8(table:singer) <-> 9(column:singer_id)
Edge: 8(table:singer) <-> 10(column:name)
Edge: 8(table:singer) <-> 11(column:country)
Edge: 8(table:singer) <-> 12(column:song_name)
Edge: 8(table:singer) <-> 13(column:song_release_year)
Edge: 8(table:singer) <-> 14(column:age)
Edge: 8(table:singer) <-> 15(column:is_male)
Edge: 8(table:singer) <-> 25(question:How many singers do we have?)
Edge: 8(table:singer) <-> 26(question:What is the total number of singers?)
Edge: 8(table:singer) <-> 27(question:Show name, country, age for all singers ordered by age from the oldest to the youngest.)
Edge: 8(table:singer) <-> 28(question:What are the names, countries, and ages for every singer in descending order of age?)
Edge: 8(table:singer) <-> 29(question:What is the average, minimum, and maximum age of all singers from France?)
Edge: 8(table:singer) <-> 30(question:What is the average, minimum, and maximum age for all French singers?)
Edge: 8(table:singer) <-> 31(question:Show the name and the release year of the song by the youngest singer.)
Edge: 8(table:singer) <-> 32(question:What are the names and release years for all the songs of the youngest singer?)
Edge: 8(table:singer) <-> 33(question:What are all distinct countries where singers above age 20 are from?)
Edge: 8(table:singer) <-> 34(question:What are  the different countries with singers above age 20?)
Edge: 8(table:singer) <-> 35(question:Show all countries and the number of singers in each country.)
Edge: 8(table:singer) <-> 36(question:How many singers are from each country?)
Edge: 8(table:singer) <-> 37(question:List all song names by singers above the average age.)
Edge: 8(table:singer) <-> 38(question:What are all the song names by singers who are older than average?)
Edge: 8(table:singer) <-> 55(question:Show countries where a singer above age 40 and a singer below 30 are from.)
Edge: 8(table:singer) <-> 60(question:List singer names and number of concerts for each singer.)
Edge: 8(table:singer) <-> 61(question:What are the names of the singers and number of concerts for each person?)
Edge: 8(table:singer) <-> 62(question:List all singer names in concerts in year 2014.)
Edge: 8(table:singer) <-> 63(question:What are the names of the singers who performed in a concert in 2014?)
Edge: 8(table:singer) <-> 64(question:what is the name and nation of the singer who have a song having 'Hey' in its name?)
Edge: 8(table:singer) <-> 65(question:What is the name and country of origin of every singer who has a song with the word 'Hey' in its title?)
Edge: 9(column:singer_id) <-> 24(column:singer_id)
Edge: 10(column:name) <-> 27(question:Show name, country, age for all singers ordered by age from the oldest to the youngest.)
Edge: 10(column:name) <-> 28(question:What are the names, countries, and ages for every singer in descending order of age?)
Edge: 10(column:name) <-> 39(question:Show location and name for all stadiums with a capacity between 5000 and 10000.)
Edge: 10(column:name) <-> 40(question:What are the locations and names of all stations with capacity between 5000 and 10000?)
Edge: 10(column:name) <-> 43(question:What is the name and capacity for the stadium with highest average attendance?)
Edge: 10(column:name) <-> 44(question:What is the name and capacity for the stadium with the highest average attendance?)
Edge: 10(column:name) <-> 47(question:Show the stadium name and the number of concerts in each stadium.)
Edge: 10(column:name) <-> 48(question:For each stadium, how many concerts play there?)
Edge: 10(column:name) <-> 49(question:Show the stadium name and capacity with most number of concerts in year 2014 or after.)
Edge: 10(column:name) <-> 50(question:What is the name and capacity of the stadium  with the most concerts after 2013?)
Edge: 10(column:name) <-> 53(question:Show the stadium names without any concert.)
Edge: 10(column:name) <-> 54(question:What are the names of the stadiums without any concerts?)
Edge: 10(column:name) <-> 56(question:Show names for all stadiums except for stadiums having a concert in year 2014.)
Edge: 10(column:name) <-> 57(question:What are the names of all stadiums that did not have a concert in 2014?)
Edge: 10(column:name) <-> 60(question:List singer names and number of concerts for each singer.)
Edge: 10(column:name) <-> 61(question:What are the names of the singers and number of concerts for each person?)
Edge: 10(column:name) <-> 62(question:List all singer names in concerts in year 2014.)
Edge: 10(column:name) <-> 63(question:What are the names of the singers who performed in a concert in 2014?)
Edge: 10(column:name) <-> 64(question:what is the name and nation of the singer who have a song having 'Hey' in its name?)
Edge: 10(column:name) <-> 65(question:What is the name and country of origin of every singer who has a song with the word 'Hey' in its title?)
Edge: 10(column:name) <-> 66(question:Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015.)
Edge: 10(column:name) <-> 67(question:What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?)
Edge: 11(column:country) <-> 27(question:Show name, country, age for all singers ordered by age from the oldest to the youngest.)
Edge: 11(column:country) <-> 28(question:What are the names, countries, and ages for every singer in descending order of age?)
Edge: 11(column:country) <-> 29(question:What is the average, minimum, and maximum age of all singers from France?)
Edge: 11(column:country) <-> 30(question:What is the average, minimum, and maximum age for all French singers?)
Edge: 11(column:country) <-> 33(question:What are all distinct countries where singers above age 20 are from?)
Edge: 11(column:country) <-> 34(question:What are  the different countries with singers above age 20?)
Edge: 11(column:country) <-> 35(question:Show all countries and the number of singers in each country.)
Edge: 11(column:country) <-> 36(question:How many singers are from each country?)
Edge: 11(column:country) <-> 55(question:Show countries where a singer above age 40 and a singer below 30 are from.)
Edge: 11(column:country) <-> 64(question:what is the name and nation of the singer who have a song having 'Hey' in its name?)
Edge: 11(column:country) <-> 65(question:What is the name and country of origin of every singer who has a song with the word 'Hey' in its title?)
Edge: 12(column:song_name) <-> 31(question:Show the name and the release year of the song by the youngest singer.)
Edge: 12(column:song_name) <-> 32(question:What are the names and release years for all the songs of the youngest singer?)
Edge: 12(column:song_name) <-> 37(question:List all song names by singers above the average age.)
Edge: 12(column:song_name) <-> 38(question:What are all the song names by singers who are older than average?)
Edge: 12(column:song_name) <-> 64(question:what is the name and nation of the singer who have a song having 'Hey' in its name?)
Edge: 12(column:song_name) <-> 65(question:What is the name and country of origin of every singer who has a song with the word 'Hey' in its title?)
Edge: 13(column:song_release_year) <-> 31(question:Show the name and the release year of the song by the youngest singer.)
Edge: 13(column:song_release_year) <-> 32(question:What are the names and release years for all the songs of the youngest singer?)
Edge: 14(column:age) <-> 27(question:Show name, country, age for all singers ordered by age from the oldest to the youngest.)
Edge: 14(column:age) <-> 28(question:What are the names, countries, and ages for every singer in descending order of age?)
Edge: 14(column:age) <-> 29(question:What is the average, minimum, and maximum age of all singers from France?)
Edge: 14(column:age) <-> 30(question:What is the average, minimum, and maximum age for all French singers?)
Edge: 14(column:age) <-> 31(question:Show the name and the release year of the song by the youngest singer.)
Edge: 14(column:age) <-> 32(question:What are the names and release years for all the songs of the youngest singer?)
Edge: 14(column:age) <-> 33(question:What are all distinct countries where singers above age 20 are from?)
Edge: 14(column:age) <-> 34(question:What are  the different countries with singers above age 20?)
Edge: 14(column:age) <-> 37(question:List all song names by singers above the average age.)
Edge: 14(column:age) <-> 38(question:What are all the song names by singers who are older than average?)
Edge: 14(column:age) <-> 55(question:Show countries where a singer above age 40 and a singer below 30 are from.)
Edge: 16(table:concert) <-> 17(column:concert_id)
Edge: 16(table:concert) <-> 18(column:concert_name)
Edge: 16(table:concert) <-> 19(column:theme)
Edge: 16(table:concert) <-> 20(column:stadium_id)
Edge: 16(table:concert) <-> 21(column:year)
Edge: 16(table:concert) <-> 45(question:How many concerts are there in year 2014 or 2015?)
Edge: 16(table:concert) <-> 46(question:How many concerts occurred in 2014 or 2015?)
Edge: 16(table:concert) <-> 47(question:Show the stadium name and the number of concerts in each stadium.)
Edge: 16(table:concert) <-> 48(question:For each stadium, how many concerts play there?)
Edge: 16(table:concert) <-> 49(question:Show the stadium name and capacity with most number of concerts in year 2014 or after.)
Edge: 16(table:concert) <-> 50(question:What is the name and capacity of the stadium  with the most concerts after 2013?)
Edge: 16(table:concert) <-> 51(question:Which year has most number of concerts?)
Edge: 16(table:concert) <-> 52(question:What is the year that had the most concerts?)
Edge: 16(table:concert) <-> 53(question:Show the stadium names without any concert.)
Edge: 16(table:concert) <-> 54(question:What are the names of the stadiums without any concerts?)
Edge: 16(table:concert) <-> 56(question:Show names for all stadiums except for stadiums having a concert in year 2014.)
Edge: 16(table:concert) <-> 57(question:What are the names of all stadiums that did not have a concert in 2014?)
Edge: 16(table:concert) <-> 58(question:Show the name and theme for all concerts and the number of singers in each concert.)
Edge: 16(table:concert) <-> 59(question:What are the names, themes, and number of singers for each and every concert?)
Edge: 16(table:concert) <-> 62(question:List all singer names in concerts in year 2014.)
Edge: 16(table:concert) <-> 63(question:What are the names of the singers who performed in a concert in 2014?)
Edge: 16(table:concert) <-> 66(question:Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015.)
Edge: 16(table:concert) <-> 67(question:What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?)
Edge: 16(table:concert) <-> 68(question:Find the number of concerts happened in the stadium with the highest capacity.)
Edge: 16(table:concert) <-> 69(question:What are the number of concerts that occurred in the stadium with the largest capacity?)
Edge: 17(column:concert_id) <-> 23(column:concert_id)
Edge: 18(column:concert_name) <-> 58(question:Show the name and theme for all concerts and the number of singers in each concert.)
Edge: 18(column:concert_name) <-> 59(question:What are the names, themes, and number of singers for each and every concert?)
Edge: 19(column:theme) <-> 58(question:Show the name and theme for all concerts and the number of singers in each concert.)
Edge: 19(column:theme) <-> 59(question:What are the names, themes, and number of singers for each and every concert?)
Edge: 20(column:stadium_id) <-> 47(question:Show the stadium name and the number of concerts in each stadium.)
Edge: 20(column:stadium_id) <-> 48(question:For each stadium, how many concerts play there?)
Edge: 20(column:stadium_id) <-> 49(question:Show the stadium name and capacity with most number of concerts in year 2014 or after.)
Edge: 20(column:stadium_id) <-> 50(question:What is the name and capacity of the stadium  with the most concerts after 2013?)
Edge: 20(column:stadium_id) <-> 53(question:Show the stadium names without any concert.)
Edge: 20(column:stadium_id) <-> 54(question:What are the names of the stadiums without any concerts?)
Edge: 20(column:stadium_id) <-> 56(question:Show names for all stadiums except for stadiums having a concert in year 2014.)
Edge: 20(column:stadium_id) <-> 57(question:What are the names of all stadiums that did not have a concert in 2014?)
Edge: 20(column:stadium_id) <-> 66(question:Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015.)
Edge: 20(column:stadium_id) <-> 67(question:What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?)
Edge: 20(column:stadium_id) <-> 68(question:Find the number of concerts happened in the stadium with the highest capacity.)
Edge: 20(column:stadium_id) <-> 69(question:What are the number of concerts that occurred in the stadium with the largest capacity?)
Edge: 21(column:year) <-> 45(question:How many concerts are there in year 2014 or 2015?)
Edge: 21(column:year) <-> 46(question:How many concerts occurred in 2014 or 2015?)
Edge: 21(column:year) <-> 49(question:Show the stadium name and capacity with most number of concerts in year 2014 or after.)
Edge: 21(column:year) <-> 50(question:What is the name and capacity of the stadium  with the most concerts after 2013?)
Edge: 21(column:year) <-> 51(question:Which year has most number of concerts?)
Edge: 21(column:year) <-> 52(question:What is the year that had the most concerts?)
Edge: 21(column:year) <-> 56(question:Show names for all stadiums except for stadiums having a concert in year 2014.)
Edge: 21(column:year) <-> 57(question:What are the names of all stadiums that did not have a concert in 2014?)
Edge: 21(column:year) <-> 62(question:List all singer names in concerts in year 2014.)
Edge: 21(column:year) <-> 63(question:What are the names of the singers who performed in a concert in 2014?)
Edge: 21(column:year) <-> 66(question:Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015.)
Edge: 21(column:year) <-> 67(question:What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?)
Edge: 22(table:singer_in_concert) <-> 23(column:concert_id)
Edge: 22(table:singer_in_concert) <-> 24(column:singer_id)
Edge: 22(table:singer_in_concert) <-> 58(question:Show the name and theme for all concerts and the number of singers in each concert.)
Edge: 22(table:singer_in_concert) <-> 59(question:What are the names, themes, and number of singers for each and every concert?)
Edge: 22(table:singer_in_concert) <-> 60(question:List singer names and number of concerts for each singer.)
Edge: 22(table:singer_in_concert) <-> 61(question:What are the names of the singers and number of concerts for each person?)
Edge: 22(table:singer_in_concert) <-> 62(question:List all singer names in concerts in year 2014.)
Edge: 22(table:singer_in_concert) <-> 63(question:What are the names of the singers who performed in a concert in 2014?)
Edge: 23(column:concert_id) <-> 58(question:Show the name and theme for all concerts and the number of singers in each concert.)
Edge: 23(column:concert_id) <-> 59(question:What are the names, themes, and number of singers for each and every concert?)
Edge: 23(column:concert_id) <-> 62(question:List all singer names in concerts in year 2014.)
Edge: 23(column:concert_id) <-> 63(question:What are the names of the singers who performed in a concert in 2014?)
Edge: 24(column:singer_id) <-> 60(question:List singer names and number of concerts for each singer.)
Edge: 24(column:singer_id) <-> 61(question:What are the names of the singers and number of concerts for each person?)
Edge: 24(column:singer_id) <-> 62(question:List all singer names in concerts in year 2014.)
Edge: 24(column:singer_id) <-> 63(question:What are the names of the singers who performed in a concert in 2014?)
======================================================================
Number of graphs in data/schema_linking_graph_dataset/link_level_graph_dataset/spider_dev_link_level_graph.pt: 20
```
