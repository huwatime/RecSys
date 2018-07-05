from RecSysBaseLine import RecSysBaseLine
from EvaSys import EvaSys

rec_sys = RecSysBaseLine()
eva_sys = EvaSys()
eva_sys.load_split_ratings(
        ["testcase_ratings_1.csv",
         "testcase_ratings_2.csv",
         "testcase_ratings_3.csv"])
result = eva_sys.evaluate(rec_sys, [1, 3, 5])
