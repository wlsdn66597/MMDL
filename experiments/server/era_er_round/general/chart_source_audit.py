"""Verify the held-out ChartQA answers against independent readings of the source images."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

# ID: (answer observed at the indicated plot label, locating witness)
OBSERVED={
    "two_col_83211":("23.6","Japan has a labeled 23.6% bar, below Costa Rica at 27.8%."),
    "75012880000273":("81.71","Of three bars, the middle one is Lower secondary, 81.71%."),
    "multi_col_60285":("2.5","Blue Spring 2018: Suzuki is the minimum at 2.5; dark Autumn minimum is 2.4."),
    "multi_col_20545":("7.94","The blue Agriculture portion of the 2015 bar is labeled 7.94%."),
    "multi_col_988":("37.3","In sexy attire, the bright-blue 13–20 group is 37.3%."),
    "3861":("16","Identity theft row: very 66, somewhat 17, not concerned 16."),
    "multi_col_20200":("18","2015 stand-alone air conditioning (bright blue) is 18%; central is 39%."),
    "multi_col_101153":("94.53","Gray Total at 2016 is 94.53; female 49.01, male 45.52."),
    "2676":("76","France has a blue Pas d'accord bar of 76."),
    "multi_col_60050":("716.55","Imported beer (bright blue, lower segment) at 2020 is 716.55; dark domestic is 798.68."),
    "multi_col_40743":("59","1990 (blue) sorts as 53,59,68,75: second lowest is 59."),
    "multi_col_60854":("35.6","In the only displayed below-50 group, 10–49 employees, the 2019 purple bar is 35.6."),
    "multi_col_60860":("13.65","2015 middle Industry segment is labeled 13.65%."),
    "multi_col_40845":("21","The dark High priority segments have their minimum at Japan, 21%."),
    "two_col_62140":("181","Pizza Hut 765, KFC 181, Taco Bell 69."),
    "multi_col_60322":("13","18–29 group's bottom Very interested slice is 13%."),
    "multi_col_60462":("42","The tallest bar is Sometimes, Republican, 42%."),
    "8527":("49","August 2016 first Very segment is 49. The source table has defects in other years."),
    "two_col_42774":("931.08","FY 2018 bar is labeled 931.08 billion rupees."),
    "multi_col_60987":("8.1","2005 Denver dark viewers bar is 8.1 million; rating is 4.9."),
    "50904014001027":("58.79","Cyprus bar is labeled 58.79%."),
    "91577275004279":("1.92","The Fast Track projection is 1.92 children per woman, compared with 2.11 for GET."),
    "multi_col_100785":("33","White row: female presidents are 33%, administrators 55%."),
    "multi_col_100081":("73","In 2013 Everyday is 57%, Often/Sometimes is 16%; the sum is 73%."),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source=(HERE/"items.jsonl").read_bytes()
    records=[r for r in map(json.loads,source.decode("utf8").splitlines()) if r["source"]=="ChartQA/val-human"]
    assert len(records)==len(OBSERVED)==24
    report=[]
    for r in records:
        identifier=r["source_id"]
        value,witness=OBSERVED[identifier]
        assert Decimal(r["reference"]["native_answer"].replace("%","").strip())==Decimal(value)
        assert Decimal(r["options"][ord(r["gold"])-65])==Decimal(value)
        image=ROOT/r["image"]
        assert sha(image.read_bytes())==r["image_sha256"]
        report.append(dict(id=r["id"],source_table_sha256=r["reference"]["source_table_sha256"],
                           image_sha256=r["image_sha256"],question=r["question"],gold_letter=r["gold"],
                           observed=value,witness=witness))
    excluded={
        "4083":"Longest bar can denote the whole West stack (31) or the longest colored segment (25); both were options.",
        "multi_col_20326":"2016 can mean 2015/16 (29.7) or 2016/17 (30.4), both offered.",
        "11756":"Target Unfavorable 2014 was 64 in both. A non-target series, Favorable 2014, was 31 visually and 0 in the source table.",
        "5436":"Middle among three drawn bars is 50, while the source table duplicates that middle row.",
        "multi_col_40608":"Denmark revised post-outbreak is 1.0; the original native answer 1.5 corresponds to pre-outbreak.",
    }
    result=dict(roster_sha256=sha(source),checked=24,all_visible_target_values_match=True,
                reviewed_before_solver_outputs=True,items=report,pre_freeze_exclusions=excluded)
    path=HERE/"chart_source_audit.json"
    content=(json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode()
    if path.exists():assert content==path.read_bytes()
    else:path.write_bytes(content)
    print(len(report),sha(content),sha(source))


if __name__=="__main__":
    main()
