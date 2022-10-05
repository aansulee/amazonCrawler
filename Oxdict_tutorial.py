import requests
import json

app_id = "79a74762"
app_key = "fbbda56ba258d085e49162ed80758d43"
language = "en-gb"
word_id = "SSD"
url = "https://od-api.oxforddictionaries.com:443/api/v2/entries/" + language + "/" + word_id.lower()
r = requests.get(url, headers={"app_id": "79a74762", "app_key": "fbbda56ba258d085e49162ed80758d43"})


print("code {}\n".format(r.status_code))
print("text \n" + r.text)
print("json \n" + json.dumps(r.json()))