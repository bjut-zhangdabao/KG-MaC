from openai import OpenAI
import os
from tqdm import tqdm
import json
import re

input_path = 'short_video_KG/train.json'
output_path = 'short_video_KG/Knowledge/train_triple.json'

client = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

entity_set = set()
relation_set = set()


triple_pattern = re.compile(r'\(([^()]+?)\s*,\s*([^()]+?)\s*,\s*([^()]+?)\)')

with open(input_path, 'r', encoding='utf-8') as f_in, \
     open(output_path, 'w', encoding='utf-8') as f_out:

    lines = f_in.readlines()

    for line in tqdm(lines, desc="Processing videos", total=len(lines)):
        try:
            data = json.loads(line.strip())
            video_id = data.get("video_id")
            if not video_id:
                continue

            news_input = data.get("llm_text_prompt", "无")
            prompt = (
                f"{news_input}\n"
                "任务：你需要从给定文本中提取事实三元组，每个三元组的形式为 (实体1, 关系, 实体2/属性/事实)。请遵循以下规则："
                "1. 实体1 和 实体2 应该尽量具体、唯一、可识别，关系用简短动词或短语描述两者的关系。"
                "2. 事件类信息（如事故、灾害、新闻事件）要提取事件发生时间、地点、主体、影响等。"
                "3. 对文本中的评论、观点、推测等信息，也提取实体与其关系，如“评论内容”、“引发讨论”等。"
                "4. 对虚假信息、未经证实信息，应明确标注关系，如“无事实依据”。"
                "5. 对关键词、标题、发布时间、作者信息等结构化数据也可以提取成三元组。"
                "6. 尽量分解复杂句子，每个有明确实体关系的信息生成一条三元组。"
                "7. 保持三元组简洁、完整且互不重复。"
                "8.实体不能出现“该文本“、”评论区“、”部分用户”、“ 发布时间”和“正文内容”等字或不唯一的实体名称。务必要保证实体1和实体2的唯一性。"
                "输出格式示例：(实体1, 关系, 实体2/属性/事实)"
                "以下给出两个示例："
                "示例1：该文本标题为“要不要主动和外星人接触？霍金曾经警告过人类”，发布时间为1595779325000，发布于抖音平台，作者简介称“感谢抖音这么优秀的平台！宇宙探索，海底探索！地球生态，未解之谜！人文地理，考古发现！这里也许有你想要的答案！喜欢➕关注，每天更新！”；正文描述提到“100多颗恒星凭空消失”，引发关于是自然现象还是外星文明干预的猜测，关键词为“100多颗恒星‘离奇消失’是被外星人操控了”；评论区观点纷杂，包括对宇宙虚拟性、外星生命威胁（如“天龙星人专吃人类”）、霍金警告的讨论，以及“被黑洞吃了”“宇宙尘埃遮挡”等解释，也有用户质疑信息真实性或表达消极态度；经核实，该文本所描述的“百余颗恒星离奇消失”等内容并无科学依据。"
                "示例1提取得到的三元组：(100多颗恒星‘离奇消失’, 是被操控了, 外星人) (霍金, 曾经警告, 人类) (要不要主动和外星人接触？霍金曾经警告过人类,  发布时间是, 1595779325000) (100多颗恒星, 凭空, 消失) (消失, 引发, 关于是自然现象) (消失, 引发, 外星文明干预的猜测) (100多颗恒星‘离奇消失’, 有，对宇宙虚拟性评论) (100多颗恒星‘离奇消失’, 有，外星生命威胁评论) (100多颗恒星‘离奇消失’, 有，天龙星人专吃人类评论) (100多颗恒星‘离奇消失’, 有，霍金警告的讨论评论) (100多颗恒星‘离奇消失’, 有，被黑洞吃了评论) (100多颗恒星‘离奇消失’, 有，宇宙尘埃遮挡评论) (100多颗恒星凭空消失, 引发, 自然现象还是外星文明)"
                "示例2：该文本标题称9日浙江衢州一化工厂突发大火、浓烟四起，消防部门已赶赴现场，并感慨“多事之秋”；正文发布于抖音账号1332395178，将事件错误关联至“徐州化工厂爆炸”，声称有毒烟雾飘至上饶市，导致鸟类死亡，呼吁公众佩戴口罩做好防护；关键词为“119 徐州化工厂爆炸”，作者简介为“谢谢你的关注，知足常乐”，发布时间为2020年11月9日（时间戳1604949052000），评论内容为“转发”及“美女晚上好，今年真是不平凡的一年，好好保重”；经核实，文中所述“徐州化工厂爆炸”及烟雾影响上饶等情况均无事实依据"
                "示例2提取得到的三元组：(浙江衢州一化工厂,突发, 大火) (浙江衢州一化工厂, 发布抖音账号, 1332395178) (119徐州化工厂爆炸, 声称有, 毒烟) (毒烟, 导致, 鸟类死亡) (119 徐州化工厂爆炸, 呼吁, 公众) (公众, 佩戴, 口罩) (公众, 做好, 防护) (119 徐州化工厂爆炸, 发布时间为, 1604949052000) (119 徐州化工厂爆炸, 有, 转发评论) (119 徐州化工厂爆炸, 有, 美女晚上好，今年真是不平凡的一年，好好保重评论) (徐州化工厂爆炸, 无, 事实依据) (烟雾影响上饶等情况, 无, 事实依据)"

            )

            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个三元组提取器"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            content = response.choices[0].message.content

            triples = triple_pattern.findall(content)

            for subject, predicate, obj in triples:
                subject = subject.strip().replace('“', '').replace('”', '').replace('‘', '').replace('’', '').replace('《', '').replace('》', '').replace('！', '').replace('，', '').replace('。', '').replace('*', '').replace('\n', '').replace('\\', '').replace('"', '')
                predicate = predicate.strip().replace('“', '').replace('”', '').replace('‘', '').replace('’', '').replace('《', '').replace('》', '').replace('！', '').replace('，', '').replace('。', '').replace('*', '').replace('\n', '').replace('\\', '').replace('"', '')
                obj = obj.strip().replace('“', '').replace('”', '').replace('‘', '').replace('’', '').replace('《', '').replace('》', '').replace('！', '').replace('，', '').replace('。', '').replace('*', '').replace('\n', '').replace('\\', '').replace('"', '')
                if subject:
                    entity_set.add(subject)
                if obj:
                    entity_set.add(obj)
                if predicate:
                    relation_set.add(predicate)

                triple_json = {
                    "head": subject.strip(),
                    "relation": predicate.strip(),
                    "tail": obj.strip()#,
                    #"video_id": video_id
                }
                with open("short_video_KG/Knowledge/entities.txt", "w", encoding="utf-8") as f:
                    for ent in entity_set:
                        f.write(ent + "\n")

                with open("short_video_KG/Knowledge/relations.txt", "w", encoding="utf-8") as f:
                    for rel in relation_set:
                        f.write(rel + "\n")
                f_out.write(json.dumps(triple_json, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"{data.get('video_id', '未知ID')} 处理失败: {e}")
