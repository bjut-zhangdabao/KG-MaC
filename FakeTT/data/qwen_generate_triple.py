from openai import OpenAI
import os
from tqdm import tqdm
import json
import re

input_path = 'short_video_KG/Knowledge/train.json'
output_path = 'short_video_KG/Knowledge/train_triple.json'

client = OpenAI(
    api_key="sk-fc0326c6",
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

            # 构造输入信息
            news_input = data.get("llm_text_prompt", "Nothing")
            # 提取三元组 prompt
            prompt = (
                f"{news_input}\n"
                "Task: Extract factual triples from the given text. Each triple should follow the format (Entity1, Relation, Entity2/Attribute/Fact). Please follow these rules:"
                "1. Entity1 and Entity2 should be as specific, unique, and identifiable as possible. The relation should be a short verb or phrase describing the relationship between them."
                "2. For event-related information (such as accidents, disasters, or news events), extract key elements including event time, location, participants, and impact."
                "3. For comments, opinions, or speculative statements in the text, also extract the corresponding entity–relation triples. Ensure that both entities and relations are specific, unique, and clearly identifiable.Avoid using vague references or pronouns, such as “this message,” “it,” “this,” “that,” “these,” “those,” or any similar ambiguous terms. All entities must be expressed with explicit and unambiguous names to ensure clarity and uniqueness."
                "4. For false or unverified information, clearly mark the relationship, such as “is false information” or “has no factual basis”."
                "5. Structured metadata such as keywords, title, publication time, and author information can also be extracted as triples."
                "6.Decompose complex sentences as much as possible, generating one triple for each clear entity relationship."
                "7. Ensure that triples are concise, complete, and non-duplicated."
                "8.Entities must not include vague or non-unique terms such as “the text,” “comment section,” “some users,” “publication time,” or “main content.” Ensure that Entity1 and Entity2 are uniquely identifiable."
                "Output format example: (Entity1, Relation, Entity2/Attribute/Fact)"
                "The following is an example: "
                "Eeample1: The viral claim of an “alien boarding UFO in Romania” is a fictional story created by the content creator “Mr. Shiver,” as indicated by the label “STORIES TOLD BY MR SHIVER! ”. Posted on Reddit with sensational hashtags like uap, aliens, and scary, the video—uploaded at timestamp 1673814633000 (July 15, 2022)—is part of a themed entertainment series, not factual reporting. In contrast, the report about a wind turbine catching fire near Crowell, Texas. Documented with descriptive detail—including large smoke rings—and published at timestamp 1658601552000 (July 22, 2022), it originates from a trusted news source committed to original, reliable journalism, and has been corroborated by user verification."
                "Triples extracted from Example 1: (Alien boarding UFO in Romania claim, is classified as, fictional story) (Alien boarding UFO in Romania claim, created by, Mr. Shiver) (STORIES TOLD BY MR SHIVER, indicates, fictional content label) (Alien boarding UFO in Romania video, posted on, Reddit) (Alien boarding UFO in Romania video, includes hashtags, uap) (Alien boarding UFO in Romania video, includes hashtags, aliens) (Alien boarding UFO in Romania video, includes hashtags, scary) (Alien boarding UFO in Romania video, uploaded at timestamp, 1673814633000) (1673814633000, corresponds to date, July 15 2022) (Alien boarding UFO in Romania video, belongs to, themed entertainment series) (Alien boarding UFO in Romania video, is not, factual reporting) (Wind turbine near Crowell Texas, caught fire during, severe storm) (Wind turbine fire near Crowell Texas event, includes detail, large smoke rings) (Wind turbine fire near Crowell Texas event, published at timestamp, 1658601552000) (1658601552000, corresponds to date, July 22 2022) (Wind turbine fire near Crowell Texas report, originates from, trusted news source) (Wind turbine fire near Crowell Texas event, corroborated by, user verification)"

            )

            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {"role": "system", "content": "You are a strict fact triple extractor that extracts precise and unambiguous triples in the format (Entity1, Relation, Entity2/Attribute/Fact) from the given text."},
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
