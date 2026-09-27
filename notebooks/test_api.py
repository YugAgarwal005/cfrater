import requests, json

resp = requests.get('https://codeforces.com/api/problemset.problems', timeout=15)
data = resp.json()
if data['status'] == 'OK':
    problems = data['result']['problems']
    rated = [p for p in problems if p.get('rating')]
    print(f'API OK! Total problems: {len(problems)}  Rated: {len(rated)}')
    for p in rated[:3]:
        cid = p['contestId']
        idx = p['index']
        name = p['name']
        rating = p['rating']
        tags = p['tags']
        print(f'  [{cid}{idx}] {name} -- Rating: {rating} -- Tags: {tags}')
else:
    print('API error:', data['status'])
