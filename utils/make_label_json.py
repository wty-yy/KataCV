# -*- coding: utf-8 -*-
'''
@File    : make_label_json.py
@Time    : 2023/09/08 21:29:48
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
当前文件用于将dataset中的类别和序号对应起来，默认按照字典序从小到大编号，dataset的文件结构如下：

main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
'''
from pathlib import Path
import json, argparse
from pathlib import Path

path_origin_dataset = Path("/home/wty/Coding/GitHub/replicate-papers/_DEBUG/datasets/imagenet")
path_logs = Path.cwd().joinpath("logs")

def write_json(path, obj):
    with open(path, "w") as file:
        json.dump(obj, file, indent=4)

def parse_args():
    cvt2path = lambda x: Path(x)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-origin-dataset", type=cvt2path, default=path_origin_dataset,
        help="the path of the original dataset (JPEG)")
    parser.add_argument("--path-logs", type=cvt2path, default=path_logs,
        help="the path for saving the json logs")
    parser.add_argument("--subfolder-name", type=str, default="train",
        help="the subfolder name of the origin dataset to find nameid, such as 'train' or 'val'")
    args = parser.parse_args()
    args.path_subdataset = args.path_origin_dataset.joinpath(args.subfolder_name)
    assert(args.path_subdataset.exists())
    args.path_logs.mkdir(exist_ok=True)
    return args

if __name__ == '__main__':
    args = parse_args()

    label_count = 0
    nameid2label, label2nameid = {}, {}
    
    for dir in sorted(Path(args.path_subdataset).iterdir()):
        nameid2label[dir.name] = label_count
        label2nameid[label_count] = dir.name
        label_count += 1
    print(f"Find {label_count} class, start writing json...")
    write_json(args.path_logs.joinpath("nameid2label.json"), nameid2label)
    write_json(args.path_logs.joinpath("label2nameid.json"), label2nameid)
    print("Finish!")
        