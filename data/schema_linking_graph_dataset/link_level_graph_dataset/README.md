Link-level graph datasets **(with edge types)**: for each dataset, the number of generated graphs equals to the number of databases

Inspection Example: **(with edge types)**

```
=== Inspecting Link-level Graph: 1 ===

Graph Structure:
Data(x=[68, 384], edge_index=[2, 159], edge_type=[159], node_names=[68], node_types=[68], database_name='farm')

Graph Statistics:
Number of nodes: 68
Number of edges: 159
Average degree: 4.68

Node Types Distribution:
  - table: 4
  - column: 24
  - question: 40

Node Information:
Node 0: city (table)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.05124494  0.07561283 -0.03347935]
Node 1: city_id (column)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.06142806  0.05627484 -0.03504805]
Node 2: official_name (column)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.00220272  0.0438046  -0.03044264]
Node 3: status (column)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.04682998  0.03798341 -0.01913936]
Node 4: area_km_2 (column)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.13977867  0.02200752 -0.00205663]
Node 5: population (column)
  - Embedding dimension: 384
  - Embedding (first three values): [0.10955072 0.03621763 0.01128592]
Node 6: census_ranking (column)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.07782522 -0.0571366   0.05478874]
Node 7: farm (table)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.047833    0.01996859 -0.04276702]
Node 8: farm_id (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.01950377  0.05375429 -0.09353965]
Node 9: year (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.04823715  0.04531972 -0.02209476]
Node 10: total_horses (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.0183107   0.02816561 -0.0088063 ]
Node 11: working_horses (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-6.5578118e-02  2.1138525e-02 -8.5200763e-06]
Node 12: total_cattle (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.00147573 -0.02448227 -0.04570609]
Node 13: oxen (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.05226019  0.00130659 -0.00944015]
Node 14: bulls (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.01240376  0.00012637 -0.02257722]
Node 15: cows (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.00310705 -0.03806737 -0.03446712]
Node 16: pigs (column)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.04408747  0.01025789 -0.03626276]
Node 17: sheep_and_goats (column)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.00739918 -0.00837085 -0.03624145]
Node 18: farm_competition (table)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.05185692  0.01254436 -0.0501252 ]
Node 19: competition_id (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.03704238  0.05336049 -0.06274432]
Node 20: year (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.04428472  0.04907979 -0.02024836]
Node 21: theme (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.03129625  0.06380371 -0.03426659]
Node 22: host_city_id (column)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.04824004  0.06001043 -0.03425519]
Node 23: hosts (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.01938409 -0.00307317 -0.02924285]
Node 24: competition_record (table)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.0489568   0.03469435 -0.0637343 ]
Node 25: competition_id (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.01582305  0.07341887 -0.06470732]
Node 26: farm_id (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.0239404   0.05818008 -0.06721161]
Node 27: rank (column)
  - Embedding dimension: 384
  - Embedding (first three values): [-0.03219736  0.00660436 -0.03280712]
Node 28: How many farms are there? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.08600319 -0.04004152 -0.03037734]
Node 29: Count the number of farms. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.06604051  0.02858084 -0.01840961]
Node 30: List the total number of horses on farms in ascending order. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [0.04206895 0.01083966 0.04797867]
Node 31: What is the total horses record for each farm, sorted ascending? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.01895342 -0.02374465  0.01366386]
Node 32: What are the hosts of competitions whose theme is not "Aliens"? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.04620636  0.02971147 -0.02234699]
Node 33: Return the hosts of competitions for which the theme is not Aliens? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.0448561   0.04822252 -0.00044475]
Node 34: What are the themes of farm competitions sorted by year in ascending order? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.02893269  0.05370421 -0.003302  ]
Node 35: Return the themes of farm competitions, sorted by year ascending. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [0.04922026 0.08506925 0.02959446]
Node 36: What is the average number of working horses of farms with more than 5000 total number of horses? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.06283045 -0.00602563  0.00278177]
Node 37: Give the average number of working horses on farms with more than 5000 total horses. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.06656298 -0.00118035  0.04104197]
Node 38: What are the maximum and minimum number of cows across all farms. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.09109645 -0.12232249 -0.05025797]
Node 39: Return the maximum and minimum number of cows across all farms. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.06930534 -0.02844496 -0.01833218]
Node 40: How many different statuses do cities have? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.1324875  -0.01627665 -0.01589424]
Node 41: Count the number of different statuses. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.0695113   0.05550572 -0.04619663]
Node 42: List official names of cities in descending order of population. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.09813131 -0.01247428  0.03610982]
Node 43: What are the official names of cities, ordered descending by population? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.09433593 -0.0012333   0.0402232 ]
Node 44: List the official name and status of the city with the largest population. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.11763192 -0.0291768  -0.00777099]
Node 45: What is the official name and status of the city with the most residents? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.15019546 -0.02822704 -0.01190813]
Node 46: Show the years and the official names of the host cities of competitions. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [0.04268568 0.08304355 0.00488687]
Node 47: Give the years and official names of the cities of each competition. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.05752626  0.04639842 -0.01457449]
Node 48: Show the official names of the cities that have hosted more than one competition. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.07093333 -0.01888268  0.03006928]
Node 49: What are the official names of cities that have hosted more than one competition? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.08941859 -0.03525949 -0.01195547]
Node 50: Show the status of the city that has hosted the greatest number of competitions. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [0.08902167 0.06004855 0.01914314]
Node 51: What is the status of the city that has hosted the most competitions? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.0965433   0.01784924 -0.01501407]
Node 52: Please show the themes of competitions with host cities having populations larger than 1000. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.12749735  0.07193499 -0.01125629]
Node 53: What are the themes of competitions that have corresponding host cities with more than 1000 residents? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.14312434  0.04180147 -0.01703593]
Node 54: Please show the different statuses of cities and the average population of cities with each status. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.1468133  -0.01511405  0.02545667]
Node 55: What are the statuses and average populations of each city? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.15539211 -0.00928848 -0.01283206]
Node 56: Please show the different statuses, ordered by the number of cities that have each. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.12472863 -0.01139086  0.03657299]
Node 57: Return the different statuses of cities, ascending by frequency. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [0.1185126  0.02812738 0.04793126]
Node 58: List the most common type of Status across cities. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.1581654  -0.03481952  0.02666136]
Node 59: What is the most common status across all cities? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.16492166 -0.03890157 -0.01036309]
Node 60: List the official names of cities that have not held any competition. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.11139651  0.00730438 -0.01205387]
Node 61: What are the official names of cities that have not hosted a farm competition? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.05867939  0.00465477 -0.01350787]
Node 62: Show the status shared by cities with population bigger than 1500 and smaller than 500. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [0.11314008 0.05396748 0.00583511]
Node 63: Which statuses correspond to both cities that have a population over 1500 and cities that have a population lower than 500? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.1018588   0.02317293 -0.04302656]
Node 64: Find the official names of cities with population bigger than 1500 or smaller than 500. (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.10232233  0.03030027 -0.02421252]
Node 65: What are the official names of cities that have population over 1500 or less than 500? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.11209296 -0.02050318 -0.03843223]
Node 66: Show the census ranking of cities whose status are not "Village". (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.11950885  0.00152302 -0.01626333]
Node 67: What are the census rankings of cities that do not have the status "Village"? (question)
  - Embedding dimension: 384
  - Embedding (first three values): [ 0.11615917 -0.09374676 -0.06962663]

Edge Information:
[Type: TABLE_COLUMN] Edge: 0(table:city) <-> 1(column:city_id)
[Type: TABLE_COLUMN] Edge: 0(table:city) <-> 2(column:official_name)
[Type: TABLE_COLUMN] Edge: 0(table:city) <-> 3(column:status)
[Type: TABLE_COLUMN] Edge: 0(table:city) <-> 4(column:area_km_2)
[Type: TABLE_COLUMN] Edge: 0(table:city) <-> 5(column:population)
[Type: TABLE_COLUMN] Edge: 0(table:city) <-> 6(column:census_ranking)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 40(question:How many different statuses do cities have?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 41(question:Count the number of different statuses.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 42(question:List official names of cities in descending order of population.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 43(question:What are the official names of cities, ordered descending by population?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 44(question:List the official name and status of the city with the largest population.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 45(question:What is the official name and status of the city with the most residents?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 46(question:Show the years and the official names of the host cities of competitions.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 47(question:Give the years and official names of the cities of each competition.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 48(question:Show the official names of the cities that have hosted more than one competition.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 49(question:What are the official names of cities that have hosted more than one competition?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 50(question:Show the status of the city that has hosted the greatest number of competitions.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 51(question:What is the status of the city that has hosted the most competitions?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 52(question:Please show the themes of competitions with host cities having populations larger than 1000.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 53(question:What are the themes of competitions that have corresponding host cities with more than 1000 residents?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 54(question:Please show the different statuses of cities and the average population of cities with each status.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 55(question:What are the statuses and average populations of each city?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 56(question:Please show the different statuses, ordered by the number of cities that have each.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 57(question:Return the different statuses of cities, ascending by frequency.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 58(question:List the most common type of Status across cities.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 59(question:What is the most common status across all cities?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 60(question:List the official names of cities that have not held any competition.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 61(question:What are the official names of cities that have not hosted a farm competition?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 62(question:Show the status shared by cities with population bigger than 1500 and smaller than 500.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 63(question:Which statuses correspond to both cities that have a population over 1500 and cities that have a population lower than 500?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 64(question:Find the official names of cities with population bigger than 1500 or smaller than 500.)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 65(question:What are the official names of cities that have population over 1500 or less than 500?)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 66(question:Show the census ranking of cities whose status are not "Village".)
[Type: QUESTION_REL] Edge: 0(table:city) <-> 67(question:What are the census rankings of cities that do not have the status "Village"?)
[Type: FOREIGN_KEY] Edge: 1(column:city_id) <-> 22(column:host_city_id)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 46(question:Show the years and the official names of the host cities of competitions.)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 47(question:Give the years and official names of the cities of each competition.)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 48(question:Show the official names of the cities that have hosted more than one competition.)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 49(question:What are the official names of cities that have hosted more than one competition?)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 50(question:Show the status of the city that has hosted the greatest number of competitions.)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 51(question:What is the status of the city that has hosted the most competitions?)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 52(question:Please show the themes of competitions with host cities having populations larger than 1000.)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 53(question:What are the themes of competitions that have corresponding host cities with more than 1000 residents?)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 60(question:List the official names of cities that have not held any competition.)
[Type: QUESTION_REL] Edge: 1(column:city_id) <-> 61(question:What are the official names of cities that have not hosted a farm competition?)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 42(question:List official names of cities in descending order of population.)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 43(question:What are the official names of cities, ordered descending by population?)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 44(question:List the official name and status of the city with the largest population.)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 45(question:What is the official name and status of the city with the most residents?)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 46(question:Show the years and the official names of the host cities of competitions.)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 47(question:Give the years and official names of the cities of each competition.)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 48(question:Show the official names of the cities that have hosted more than one competition.)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 49(question:What are the official names of cities that have hosted more than one competition?)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 60(question:List the official names of cities that have not held any competition.)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 61(question:What are the official names of cities that have not hosted a farm competition?)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 64(question:Find the official names of cities with population bigger than 1500 or smaller than 500.)
[Type: QUESTION_REL] Edge: 2(column:official_name) <-> 65(question:What are the official names of cities that have population over 1500 or less than 500?)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 40(question:How many different statuses do cities have?)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 41(question:Count the number of different statuses.)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 44(question:List the official name and status of the city with the largest population.)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 45(question:What is the official name and status of the city with the most residents?)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 50(question:Show the status of the city that has hosted the greatest number of competitions.)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 51(question:What is the status of the city that has hosted the most competitions?)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 54(question:Please show the different statuses of cities and the average population of cities with each status.)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 55(question:What are the statuses and average populations of each city?)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 56(question:Please show the different statuses, ordered by the number of cities that have each.)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 57(question:Return the different statuses of cities, ascending by frequency.)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 58(question:List the most common type of Status across cities.)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 59(question:What is the most common status across all cities?)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 62(question:Show the status shared by cities with population bigger than 1500 and smaller than 500.)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 63(question:Which statuses correspond to both cities that have a population over 1500 and cities that have a population lower than 500?)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 66(question:Show the census ranking of cities whose status are not "Village".)
[Type: QUESTION_REL] Edge: 3(column:status) <-> 67(question:What are the census rankings of cities that do not have the status "Village"?)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 42(question:List official names of cities in descending order of population.)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 43(question:What are the official names of cities, ordered descending by population?)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 44(question:List the official name and status of the city with the largest population.)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 45(question:What is the official name and status of the city with the most residents?)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 52(question:Please show the themes of competitions with host cities having populations larger than 1000.)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 53(question:What are the themes of competitions that have corresponding host cities with more than 1000 residents?)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 54(question:Please show the different statuses of cities and the average population of cities with each status.)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 55(question:What are the statuses and average populations of each city?)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 62(question:Show the status shared by cities with population bigger than 1500 and smaller than 500.)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 63(question:Which statuses correspond to both cities that have a population over 1500 and cities that have a population lower than 500?)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 64(question:Find the official names of cities with population bigger than 1500 or smaller than 500.)
[Type: QUESTION_REL] Edge: 5(column:population) <-> 65(question:What are the official names of cities that have population over 1500 or less than 500?)
[Type: QUESTION_REL] Edge: 6(column:census_ranking) <-> 66(question:Show the census ranking of cities whose status are not "Village".)
[Type: QUESTION_REL] Edge: 6(column:census_ranking) <-> 67(question:What are the census rankings of cities that do not have the status "Village"?)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 8(column:farm_id)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 9(column:year)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 10(column:total_horses)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 11(column:working_horses)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 12(column:total_cattle)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 13(column:oxen)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 14(column:bulls)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 15(column:cows)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 16(column:pigs)
[Type: TABLE_COLUMN] Edge: 7(table:farm) <-> 17(column:sheep_and_goats)
[Type: QUESTION_REL] Edge: 7(table:farm) <-> 28(question:How many farms are there?)
[Type: QUESTION_REL] Edge: 7(table:farm) <-> 29(question:Count the number of farms.)
[Type: QUESTION_REL] Edge: 7(table:farm) <-> 30(question:List the total number of horses on farms in ascending order.)
[Type: QUESTION_REL] Edge: 7(table:farm) <-> 31(question:What is the total horses record for each farm, sorted ascending?)
[Type: QUESTION_REL] Edge: 7(table:farm) <-> 36(question:What is the average number of working horses of farms with more than 5000 total number of horses?)
[Type: QUESTION_REL] Edge: 7(table:farm) <-> 37(question:Give the average number of working horses on farms with more than 5000 total horses.)
[Type: QUESTION_REL] Edge: 7(table:farm) <-> 38(question:What are the maximum and minimum number of cows across all farms.)
[Type: QUESTION_REL] Edge: 7(table:farm) <-> 39(question:Return the maximum and minimum number of cows across all farms.)
[Type: FOREIGN_KEY] Edge: 8(column:farm_id) <-> 26(column:farm_id)
[Type: QUESTION_REL] Edge: 10(column:total_horses) <-> 30(question:List the total number of horses on farms in ascending order.)
[Type: QUESTION_REL] Edge: 10(column:total_horses) <-> 31(question:What is the total horses record for each farm, sorted ascending?)
[Type: QUESTION_REL] Edge: 10(column:total_horses) <-> 36(question:What is the average number of working horses of farms with more than 5000 total number of horses?)
[Type: QUESTION_REL] Edge: 10(column:total_horses) <-> 37(question:Give the average number of working horses on farms with more than 5000 total horses.)
[Type: QUESTION_REL] Edge: 11(column:working_horses) <-> 36(question:What is the average number of working horses of farms with more than 5000 total number of horses?)
[Type: QUESTION_REL] Edge: 11(column:working_horses) <-> 37(question:Give the average number of working horses on farms with more than 5000 total horses.)
[Type: QUESTION_REL] Edge: 15(column:cows) <-> 38(question:What are the maximum and minimum number of cows across all farms.)
[Type: QUESTION_REL] Edge: 15(column:cows) <-> 39(question:Return the maximum and minimum number of cows across all farms.)
[Type: TABLE_COLUMN] Edge: 18(table:farm_competition) <-> 19(column:competition_id)
[Type: TABLE_COLUMN] Edge: 18(table:farm_competition) <-> 20(column:year)
[Type: TABLE_COLUMN] Edge: 18(table:farm_competition) <-> 21(column:theme)
[Type: TABLE_COLUMN] Edge: 18(table:farm_competition) <-> 22(column:host_city_id)
[Type: TABLE_COLUMN] Edge: 18(table:farm_competition) <-> 23(column:hosts)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 32(question:What are the hosts of competitions whose theme is not "Aliens"?)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 33(question:Return the hosts of competitions for which the theme is not Aliens?)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 34(question:What are the themes of farm competitions sorted by year in ascending order?)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 35(question:Return the themes of farm competitions, sorted by year ascending.)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 46(question:Show the years and the official names of the host cities of competitions.)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 47(question:Give the years and official names of the cities of each competition.)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 48(question:Show the official names of the cities that have hosted more than one competition.)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 49(question:What are the official names of cities that have hosted more than one competition?)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 50(question:Show the status of the city that has hosted the greatest number of competitions.)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 51(question:What is the status of the city that has hosted the most competitions?)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 52(question:Please show the themes of competitions with host cities having populations larger than 1000.)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 53(question:What are the themes of competitions that have corresponding host cities with more than 1000 residents?)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 60(question:List the official names of cities that have not held any competition.)
[Type: QUESTION_REL] Edge: 18(table:farm_competition) <-> 61(question:What are the official names of cities that have not hosted a farm competition?)
[Type: FOREIGN_KEY] Edge: 19(column:competition_id) <-> 25(column:competition_id)
[Type: QUESTION_REL] Edge: 20(column:year) <-> 34(question:What are the themes of farm competitions sorted by year in ascending order?)
[Type: QUESTION_REL] Edge: 20(column:year) <-> 35(question:Return the themes of farm competitions, sorted by year ascending.)
[Type: QUESTION_REL] Edge: 20(column:year) <-> 46(question:Show the years and the official names of the host cities of competitions.)
[Type: QUESTION_REL] Edge: 20(column:year) <-> 47(question:Give the years and official names of the cities of each competition.)
[Type: QUESTION_REL] Edge: 21(column:theme) <-> 32(question:What are the hosts of competitions whose theme is not "Aliens"?)
[Type: QUESTION_REL] Edge: 21(column:theme) <-> 33(question:Return the hosts of competitions for which the theme is not Aliens?)
[Type: QUESTION_REL] Edge: 21(column:theme) <-> 34(question:What are the themes of farm competitions sorted by year in ascending order?)
[Type: QUESTION_REL] Edge: 21(column:theme) <-> 35(question:Return the themes of farm competitions, sorted by year ascending.)
[Type: QUESTION_REL] Edge: 21(column:theme) <-> 52(question:Please show the themes of competitions with host cities having populations larger than 1000.)
[Type: QUESTION_REL] Edge: 21(column:theme) <-> 53(question:What are the themes of competitions that have corresponding host cities with more than 1000 residents?)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 46(question:Show the years and the official names of the host cities of competitions.)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 47(question:Give the years and official names of the cities of each competition.)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 48(question:Show the official names of the cities that have hosted more than one competition.)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 49(question:What are the official names of cities that have hosted more than one competition?)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 50(question:Show the status of the city that has hosted the greatest number of competitions.)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 51(question:What is the status of the city that has hosted the most competitions?)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 52(question:Please show the themes of competitions with host cities having populations larger than 1000.)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 53(question:What are the themes of competitions that have corresponding host cities with more than 1000 residents?)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 60(question:List the official names of cities that have not held any competition.)
[Type: QUESTION_REL] Edge: 22(column:host_city_id) <-> 61(question:What are the official names of cities that have not hosted a farm competition?)
[Type: QUESTION_REL] Edge: 23(column:hosts) <-> 32(question:What are the hosts of competitions whose theme is not "Aliens"?)
[Type: QUESTION_REL] Edge: 23(column:hosts) <-> 33(question:Return the hosts of competitions for which the theme is not Aliens?)
[Type: TABLE_COLUMN] Edge: 24(table:competition_record) <-> 25(column:competition_id)
[Type: TABLE_COLUMN] Edge: 24(table:competition_record) <-> 26(column:farm_id)
[Type: TABLE_COLUMN] Edge: 24(table:competition_record) <-> 27(column:rank)
======================================================================
Number of graphs in data/schema_linking_graph_dataset/link_level_graph_dataset/sentence_transformer/spider_train_link_level_graph.pt: 146
```
