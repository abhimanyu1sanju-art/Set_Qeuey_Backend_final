from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.environ['MONGODB_URI'])
db = client[os.environ['DATABASE_NAME']]
scenes = list(db.satellite_scenes.find({"satellite": "sentinel-2"}).limit(2))
if len(scenes) == 2:
    print(scenes[0]["scene_id"])
    print(scenes[1]["scene_id"])
