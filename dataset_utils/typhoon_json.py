from datetime import datetime
from functools import reduce
import os
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.dates
from collections import Counter

from collections import namedtuple

Range = namedtuple("Range", ["start", "end"])


def cnt_typhoon_item_class_number_from_jsons(years=[2019, 2020], json_dir="."):
    info_dict = {"datetime": [], "lon": [], "lat": [], "maxSpeed": [], "type": []}
    cnt = 0
    typhoon_type_speed_dict = {str(i): [] for i in range(6)}
    typhoon_type_info_dict = {str(i): [] for i in range(6)}
    for year in years:
        json_year_dir = os.path.join(json_dir, "inter_typhoon_" + str(year))
        json_fns = glob.glob(json_year_dir + "/*.json")
        assert len(json_fns) > 0, "no json file found in " + json_year_dir
        for json_fn in json_fns:
            with open(json_fn, "r") as fp:
                json_items = json.load(fp)
                for item in json_items:
                    cnt += 1
                    for k, v in item.items():
                        info_dict[k].append(v)
                    typhoon_type_speed_dict[str(item["type"])].append(item["maxSpeed"])
                    typhoon_type_info_dict[str(item["type"])].append(item)
    print("total json item number:", cnt)
    info_cnt_dict = {"datetime": [], "lon": [], "lat": [], "maxSpeed": [], "type": []}
    for k, v in info_dict.items():
        info_cnt_dict[k] = Counter(v)
    for k, v in info_cnt_dict.items():
        print(k, len(v))
    for k, v in typhoon_type_speed_dict.items():
        print(k, min(v), max(v))
    # typhoon_type_info_dict 分类别台风定位及时间信息

    print("done")


def get_intersect():
    # https://stackoverflow.com/questions/47141558/how-to-the-get-the-intersections-of-multiple-periods
    def intersect(range1, range2):
        new_range = Range(max(range1.start, range2.start), min(range1.end, range2.end))
        return new_range if new_range.start < new_range.end else None

    def intersect_two(ranges1, ranges2):
        for range1 in ranges1:
            for range2 in ranges2:
                intersection = intersect(range1, range2)
                if intersection:
                    yield intersection

    def intersect_all(ranges):
        return reduce(intersect_two, ranges)

    timelines = [
        (Range(0, 11), Range(15, 20)),
        (Range(8, 16), Range(19, 25)),
        (Range(0, 10), Range(15, 22)),
    ]

    for intersection in intersect_all(timelines):
        print(intersection)

    return


def get_start_time(json_fn):
    with open(json_fn, "r") as f:
        data = json.load(f)
    start_time_str = data[0]["datetime"]
    return datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")


def merge_datetime_intervals(datetime_range_list):
    # datetime_range_list : [ [start_datetime, end_datetime] ]
    # merge intervals , https://zhuanlan.zhihu.com/p/107785293
    res = []
    res.append(datetime_range_list[0])
    for i in range(1, len(datetime_range_list)):
        curr = datetime_range_list[i]
        # res 中最后一个元素的引用
        last = res[-1]
        if curr[0] <= last[1]:
            last[1] = max(last[1], curr[1])
        else:
            res.append(curr)

    print("Before merge, len: {}".format(len(datetime_range_list)))
    print("After merge, len: {}".format(len(res)))

    return res


def plot_timerange_matplot(
    json_fns=None, is_plot=False, plot_fn="pic/datetime_range.png"
):

    print("len_json_fns: %d" % len(json_fns))

    plt.figure(figsize=(20, 5), dpi=500)

    # sort json fns by start time
    json_fns = sorted(json_fns, key=lambda x: get_start_time(x), reverse=False)

    datetime_range_list = []

    for i, json_fn in enumerate(json_fns):
        with open(json_fn, "r") as f:
            data = json.load(f)
        start_time_str = data[0]["datetime"]
        end_time_str = data[-1]["datetime"]
        start_datetime = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

        datetime_range = [start_datetime, end_datetime]
        id_for_y_axis = [i, i]

        datetime_range_list.append(datetime_range)
        dates = matplotlib.dates.date2num(datetime_range)
        plt.plot_date(dates, id_for_y_axis, linestyle="solid")

    # plot merged
    merged_datetime_range_list = merge_datetime_intervals(datetime_range_list)

    if is_plot is True:
        for item in merged_datetime_range_list:
            dates = matplotlib.dates.date2num(item)
            y_label = len(datetime_range_list) + 1
            plt.plot_date(
                dates, [y_label, y_label], linestyle="solid", fmt="o", color="b"
            )

        plt.xlabel("Time")
        plt.ylabel("Yield")
        # plt.gcf().autofmt_xdate
        plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=30))
        plt.xticks(rotation=25)
        date_format = matplotlib.dates.DateFormatter("%d-%m-%Y")
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.savefig(plot_fn)

    return merged_datetime_range_list


def dump_datatime_range(datetime_range_list, save_txt_fn=None):

    with open(save_txt_fn, "w") as f:
        for datetime_range in datetime_range_list:
            # print(datetime_range[0].strftime("%Y-%m-%d %H:%M:%S"),datetime_range[1].strftime( "%Y-%m-%d %H:%M:%S"))
            time_delta_day = (
                (datetime_range[1] - datetime_range[0]).total_seconds() / 60 / 60 / 24
            )
            line = (
                datetime_range[0].strftime("%Y-%m-%d %H:%M:%S")
                + " | "
                + datetime_range[1].strftime("%Y-%m-%d %H:%M:%S")
                + " | "
                + " Day Length: {:.2f}".format(time_delta_day)
                + "     |  \n"
            )

            f.write(line)

    return


if __name__ == "__main__":

    years = [2018, 2019, 2020, 2021]

    for year in years:
        json_fns = glob.glob("./typhoon_json/typhoon_item_SH/{}/*.json".format(year))
        plot_fn = "./typhoon_json/typhoon_item_SH/ranges_plot/{}.png".format(year)
        merged_datetime_range_list = plot_timerange_matplot(
            json_fns=json_fns, is_plot=True, plot_fn=plot_fn
        )

        txt_save_dir = "typhoon_json/typhoon_item_SH/ranges_txt/"
        dump_datatime_range(
            merged_datetime_range_list, save_txt_fn=txt_save_dir + str(year) + ".txt"
        )

    print("done...")
