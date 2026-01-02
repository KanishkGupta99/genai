from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text="""
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace std;
using namespace __gnu_pbds;

#define int long long
#define no cout << "NO" << endl;
#define yes cout << "YES" << endl;

typedef tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update> pbds;
// find_by_order(k), order_of_key(x)

int MOD = 1e9 + 7;

int binpow(int a, int b, int m) {
    a %= m;
    int res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
}

vector<int> seive(int n) {
    vector<bool> prime(n + 1, true);
    for (int p = 2; p * p <= n; p++) {
        if (prime[p] == true) {
            for (int i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }
    vector<int> p;
    for (int i = 2; i <= n; i++) {
        if (prime[i]) p.push_back(i);
    }
    return p;
}

//-----your code logic here------

void solve() {
    int n;cin>>n;
    vector<int>a(n), b(n), c(n);
    for(int i=0;i<n;i++) cin>>a[i];
    for(int i=0;i<n;i++) cin>>b[i];
    for(int i=0;i<n;i++) cin>>c[i];

    int a1=0,b1=0;

    for(int i=0;i<n;i++){
        bool f=true;
        for(int j=0;j<n;j++){
            if(a[j]>=b[(i+j)%n]){
                f=false; break;
            }
        }

        if(f) a1++;
    }

    for(int j=0;j<n;j++){
        bool f=true;
        for(int k=0;k<n;k++){
            if(b[k]>=c[(k+j)%n]){
                f=false; break;
            }
        }

        if(f) b1++;
    }

    cout<<a1*b1*n<<endl;
}

int32_t main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int t = 1;
    cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}
"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.CPP,
    chunk_size=300,
    chunk_overlap=0
)

result=splitter.split_text(text)

print(len(result))
print(result)