
import re
import json
import pdb
from futsimulator.interfaces.redisinjectors import InjectZadd
from futsimulator.interfaces.redisinjectors import InjectStr2List

host_redis = '192.168.1.48'
port_redis = 6379
main_path = "/home/mora/Documents/projects/dataseries/databento/UB/"
prefix = "UB"
suffix = "zadd"

# files = ["glbx-mdp3-20240331.tbbo.json"]


files = [
    "glbx-mdp3-20240331.tbbo.json",  "glbx-mdp3-20240407.tbbo.json", "glbx-mdp3-20240414.tbbo.json",
    "glbx-mdp3-20240401.tbbo.json",  "glbx-mdp3-20240408.tbbo.json", "glbx-mdp3-20240415.tbbo.json",
    "glbx-mdp3-20240402.tbbo.json",  "glbx-mdp3-20240409.tbbo.json", "glbx-mdp3-20240416.tbbo.json",
    "glbx-mdp3-20240403.tbbo.json",  "glbx-mdp3-20240410.tbbo.json", "glbx-mdp3-20240417.tbbo.json",
    "glbx-mdp3-20240404.tbbo.json",  "glbx-mdp3-20240411.tbbo.json", "glbx-mdp3-20240418.tbbo.json",
    "glbx-mdp3-20240405.tbbo.json",  "glbx-mdp3-20240412.tbbo.json"]

files = ["glbx-mdp3-20240410.tbbo.json"]

pattern = r'\d{8}'

for file in files:
    # Search for the pattern in the text
    match = re.search(pattern, file)

    # Extract and print the matched string
    if match:
        date_string = match.group(0)
        main_path_file = main_path + file
        list_name_redis = prefix + "_" + date_string

        # Inject the dataprices
        r_inj = InjectStr2List(host_redis, port_redis)
        list_name_redis_idx = list_name_redis + '_' + suffix
        r_inj_z = InjectZadd(host_redis, port_redis)

        f = open(main_path_file)
        for idx, info_str in enumerate(f.readlines()):

            data = json.loads(info_str)
            if float(data['price']) <0 or int(data['levels'][0]['ask_px']) > 122375000000*5:
                continue

            r_inj.inject(list_name_redis, info_str)
    
            # Inject the indexing of the dataprices
            #print(data['ts_recv'])
            score = int(data['ts_recv'])
            data = {idx:score}
            r_inj_z.inject(list_name_redis_idx, data)