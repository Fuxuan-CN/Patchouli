"""
提取 THBWiki「东方求闻史纪/{角色}」页面关键信息
"""

import requests
from bs4 import BeautifulSoup
import textwrap
from pathlib import Path

TRAIN_DATASET_PATH = Path(__file__).parent.parent / "data" / "train"
""" 训练数据集路径 """

TOUHOU_POSITION: list[str] = [
    "人类村落", "博丽神社", "香霖堂", "雾之湖", "迷途竹林", 
    "魔法森林", "妖怪之山", "中有之道", "三途河", "太阳花田",
    "大蛤蟆之池", "无名之丘", "红魔馆", "冥界", "白玉楼",
    "永远亭", "月之都", "再思之道", "无缘塚", "彼岸",
]

QIUJIN_NAMES: list[str] = [
    # 妖精
    "琪露诺", "莉莉霍瓦特", "桑尼米尔克", "露娜切露德", "斯塔萨菲雅",
    # 幽灵
    "魂魄妖梦", "露娜萨·普莉兹姆利巴", "梅露兰·普莉兹姆利巴", "莉莉卡·普莉兹姆利巴",
    # 妖怪
    "露米娅", "蕾蒂·霍瓦特洛克", "莉格露·奈特巴格", "米斯蒂娅·萝蕾拉",
    "红美铃", "梅蒂欣·梅兰可莉", "风见幽香", "八云紫",
    # 魔法使
    "爱丽丝·玛格特洛依德", "帕秋莉·诺蕾姬",
    # 妖兽
    "橙", "八云蓝", "铃仙·优昙华院·因幡", "因幡天为",
    # 兽人
    "上白泽慧音",
    # 吸血鬼
    "蕾米莉亚·斯卡蕾特", "芙兰朵露·斯卡蕾特",
    # 亡灵
    "西行寺幽幽子",
    # 天狗
    "射命丸文",
    # 鬼
    "伊吹萃香",
    # 死神
    "小野塚小町",
    # 阎魔
    "四季映姬·夜摩仙那度",
    # 英雄传
    "博丽灵梦", "雾雨魔理沙", "十六夜咲夜", "森近霖之助",
    "八意永琳", "蓬莱山辉夜", "藤原妹红"
]

MEME_DICT: dict[str, str] = {
    # 妖精组
    "琪露诺": "⑨",
    "莉莉霍瓦特": "莉莉白",
    "桑尼米尔克": "太阳精",
    "露娜切露德": "月之妖精",
    "斯塔萨菲雅": "星屑妖精",
    # 幽灵组
    "魂魄妖梦": "半人半灵",
    "露娜萨·普莉兹姆利巴": "长女·小提琴",
    "梅露兰·普莉兹姆利巴": "次女·小号",
    "莉莉卡·普莉兹姆利巴": "三女·键盘",
    # 妖怪组
    "露米娅": "小碎骨",
    "蕾蒂·霍瓦特洛克": "冻青蛙",
    "莉格露·奈特巴格": "虫虫",
    "米斯蒂娅·萝蕾拉": "夜雀烧烤",
    "红美铃": "中国",
    "梅蒂欣·梅兰可莉": "毒人偶",
    "风见幽香": "四季花田的暴君",
    "八云紫": "紫妈",
    # 魔法使
    "爱丽丝·玛格特洛依德": "七色人偶使",
    "帕秋莉·诺蕾姬": "不动的大图书馆",
    # 妖兽
    "橙": "猫车",
    "八云蓝": "隙间妖的式神",
    "铃仙·优昙华院·因幡": "月兔",
    "因幡天为": "幸运兔",
    # 兽人
    "上白泽慧音": "半兽教师",
    # 吸血鬼
    "蕾米莉亚·斯卡蕾特": "红魔馆大小姐",
    "芙兰朵露·斯卡蕾特": "二小姐",
    # 亡灵
    "西行寺幽幽子": "亡灵公主",
    # 天狗
    "射命丸文": "狗仔文",
    # 鬼
    "伊吹萃香": "西瓜",
    # 死神
    "小野塚小町": "死神的小跟班",
    # 阎魔
    "四季映姬·夜摩仙那度": "说教王",
    # 英雄传
    "博丽灵梦": "红白",
    "雾雨魔理沙": "黑白",
    "十六夜咲夜": "PAD长",
    "森近霖之助": "店长",
    "八意永琳": "月之头脑",
    "蓬莱山辉夜": "NEET姬",
    "藤原妹红": "不死鸟",
}

class DatasetCollector:
    """Collects Touhou character data from THBWiki and saves it to files."""

    TEMPLATE = textwrap.dedent("""\
        <|system|>东方求闻史纪 {name}
        {meme_name}
        {name} {en_name}

        能力：{ability}
        危险度：{danger}
        人类友好度：{friend}
        主要活动场所：{place}

        {body}
        <|end|>
    """)

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; THBWikiBot/1.0; +https://example.com/bot)"
    }
    BASE_URL = "https://thbwiki.cc/东方求闻史纪/{character_name}"

    def __init__(self, output_path: Path) -> None:
        """
        Initialize the DatasetCollector.
        
        Args:
            output_path: Path to save the collected data. If None, defaults to ../data/train/
        """
        if output_path is None:
            self.output_path = Path(__file__).parent.parent / "data" / "train"
        else:
            self.output_path = output_path
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

    def fetch_html(self, name: str) -> str:
        """Fetch HTML content for a character from THBWiki."""
        url = self.BASE_URL.format(character_name=name)
        response = requests.get(url, headers=self.HEADERS, timeout=10, allow_redirects=True)
        response.raise_for_status()
        return response.text

    def parse_data(self, html: str, character_name: str) -> dict[str, str]:
        """Parse HTML data into a structured dictionary."""
        soup = BeautifulSoup(html, "lxml")
        info = {}

        # 1. Extract infobox fields
        for tr in soup.select("table[border='0'] tr"):
            tds = tr.find_all("td")
            if len(tds) == 2:
                key = tds[0].get_text(" ", strip=True)
                val = tds[1].get_text(" ", strip=True)
                info[key] = val

        # 2. Extract main content
        body = []
        content = soup.select_one(".mw-parser-output")
        if content:
            for tag in content.find_all(["p", "ul", "ol", "div"]):
                if tag.find_parent(class_="navbox"):
                    continue
                txt = tag.get_text(" ", strip=True)
                if txt:
                    body.append(txt)

        try:
            seq = body.index(character_name)
        except ValueError:
            seq = len(body)

        clean = body[2:seq]
        info["正文"] = "\n".join(clean)

        return info

    def collect_character_data(self, name: str) -> dict[str, str]:
        """Collect data for a single character from THBWiki."""
        html = self.fetch_html(name)
        return self.parse_data(html, name)

    def save_character_data(self, name: str, data: dict[str, str]) -> bool:
        """
        Save collected character data to a file.
        
        Returns:
            bool: True if data was saved successfully, False if content was empty
        """
        body = data.get("正文", "")
        if not body:
            return False

        dataset_raw = self.TEMPLATE.format(
            name=name,
            meme_name=MEME_DICT.get(name, ""),
            en_name="",  # Could be extended to collect English names
            ability=data.get("能力", "UNKNOWN"),
            danger=data.get("危险度", "UNKNOWN"),
            friend=data.get("人类友好度", "UNKNOWN"),
            place=data.get("主要活动场所", "UNKNOWN"),
            body=body
        )

        output_file = self.output_path / f"thbwiki_{name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(dataset_raw)
        
        return True

    def collect_dataset(self, name_list: list[str] = []) -> None:
        """Collect dataset for all characters in the name list."""
        if name_list is None or len(name_list) <= 0:
            name_list = QIUJIN_NAMES

        total = len(name_list)
        for idx, name in enumerate(name_list, 1):
            print(f"Collecting data: ({idx}/{total}) {name}")
            try:
                data = self.collect_character_data(name)
                if not self.save_character_data(name, data):
                    print(f"Skipped {name} (empty content)")
            except Exception as e:
                print(f"Failed to collect data for {name}: {str(e)}")

if __name__ == "__main__":
    collector = DatasetCollector(TRAIN_DATASET_PATH)
    collector.collect_dataset()