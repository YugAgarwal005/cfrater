"""
Diagnostic: Send 3 sequential identical-format but different-content predictions
and dump exact request hashes and response values.
"""
import requests, hashlib, json, time

BASE = "http://localhost:5000"

PROBLEMS = [
    {
        "statement": "Given an array of n integers, print them in sorted order.",
        "constraints": "1 <= n <= 100\nTime limit: 1 second\nMemory limit: 256 megabytes",
        "tags": ["implementation", "sortings"],
        "input_spec": "",
        "output_spec": "",
        "solution_code": "",
    },
    {
        "statement": "You have a rooted tree with n nodes. Each node has a value. For each query (u, v), find the LCA and sum of values on the path. Use binary lifting and prefix sums on Euler tour.",
        "constraints": "1 <= n <= 2*10^5\n1 <= q <= 10^5\nTime limit: 3 seconds\nMemory limit: 512 megabytes",
        "tags": ["trees", "dfs and similar", "binary search"],
        "input_spec": "First line: n q. Next n-1 lines: edges. Next n lines: node values. Next q lines: queries.",
        "output_spec": "For each query print the sum.",
        "solution_code": "#include<bits/stdc++.h>\nusing namespace std;\nconst int MAXN=200005,LOG=18;\nvector<int> adj[MAXN];\nlong long val[MAXN],depth[MAXN],up[MAXN][LOG],psum[MAXN];\nvoid dfs(int u,int p,int d){depth[u]=d;up[u][0]=p;for(int i=1;i<LOG;i++)up[u][i]=up[up[u][i-1]][i-1];for(int v:adj[u])if(v!=p){psum[v]=psum[u]+val[v];dfs(v,u,d+1);}}\nint lca(int u,int v){if(depth[u]<depth[v])swap(u,v);int diff=depth[u]-depth[v];for(int i=0;i<LOG;i++)if((diff>>i)&1)u=up[u][i];if(u==v)return u;for(int i=LOG-1;i>=0;i--)if(up[u][i]!=up[v][i]){u=up[u][i];v=up[v][i];}return up[u][0];}\nint main(){int n,q;cin>>n>>q;...}",
    },
    {
        "statement": "Find all prime numbers up to n using Sieve of Eratosthenes, then for each prime p, compute p^2 mod 10^9+7.",
        "constraints": "1 <= n <= 10^6\nTime limit: 1 second\nMemory limit: 256 megabytes",
        "tags": ["number theory", "math"],
        "input_spec": "Single integer n.",
        "output_spec": "Print all prime squares modulo 10^9+7.",
        "solution_code": "",
    },
]

print("=" * 60)
print("DIAGNOSTIC: Sequential Prediction Test")
print("=" * 60)

for i, prob in enumerate(PROBLEMS, 1):
    body_json = json.dumps(prob)
    req_hash = hashlib.md5(body_json.encode()).hexdigest()[:8]
    
    r = requests.post(f"{BASE}/api/predict",
                      data=body_json,
                      headers={"Content-Type": "application/json",
                               "Cache-Control": "no-cache"},
                      timeout=30)
    resp = r.json()
    rating = resp.get("predicted_rating")
    demo = resp.get("_demo_mode", False)
    
    print(f"\nRequest {i}  [body-md5={req_hash}]")
    print(f"  Tags:           {prob['tags']}")
    print(f"  Statement[0:60]: {prob['statement'][:60]}")
    print(f"  -> predicted_rating: {rating}")
    print(f"  -> demo_mode:        {demo}")
    print(f"  -> all model preds:  {resp.get('model_predictions', {})}")
    
    time.sleep(0.3)

print("\n" + "=" * 60)
