#include <iostream>
#include <set>
#include <map>

using namespace std;

void mySwap(int *p1, int *p2) {
    int tmp = *p1;
    *p1 = *p2;
    *p2 = tmp;
}

void myReset(int &x) {
    x = 0;
}

void mySwapR(int &x, int &y) {
    int tmp = x;
    x = y;
    y = tmp;
}

int main() {
    int x, y;
    cin >> x >> y;
    mySwapR(x, y);
    cout << x << " " << y << endl;
    return 0;
}
