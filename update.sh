#!/usr/bin/env sh

docker exec joplin sqlite3 -header -csv /root/.config/joplin/database.sqlite "select title,body from notes limit -1 offset 1;" > result.csv
/Users/josh/miniconda3/envs/semantic/bin/python build_index.py
