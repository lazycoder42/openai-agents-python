import http.client

conn = http.client.HTTPSConnection("googlesearch-api.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "84b41d3555mshe80406780df636dp1dbfa1jsnb9164da048b2",
    'x-rapidapi-host': "googlesearch-api.p.rapidapi.com"
}

conn.request("GET", "/search?q=GDP%20of%20UK%20in%20last%2010%20years&start=1&gl=US&hl=en&lr=lang_en", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))