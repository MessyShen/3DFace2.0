#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <set>
#include <queue>
#include <cmath>
#include <dirent.h>
#include <map>
using namespace std;

const int MOD = 1000000007;
typedef long long ll;

struct Point {
    int id;
    double x, y, z;
    double u, r, v;
    Point(){}
    Point(int id, double x, 
          double y, double z) : id(id), x(x), y(y), z(z) {
        r = sqrt(x*x + z*z);
        u = acos(x / r);
        v = y;
        //printf("%d init\n", id);
    }
    double dis(Point &rhs, double sc) {
        return (u-rhs.u)*(u-rhs.u)*sc*sc 
                + (y - rhs.y)*(y - rhs.y);
    }
}fcPoints[50000], centerPoint[30000];

int fc[27000][5];
int fclen;
int cntFcPoint, cntRefPoint;
double sc;


void resample(string fcInputFc, string fcInputXYZ, string faceReference) {
    ifstream input_fc(fcInputFc.c_str());
    ifstream input_pt(fcInputXYZ.c_str());
    // ifstream ref_fc("faceReference.fc", ios::in);
    ifstream ref_pt(faceReference.c_str());
    string outFile = "./Resampled/" + fcInputXYZ;
    ofstream resampled(outFile.c_str());
    int num;
    int i = 0, j = 0;
    while (input_fc >> num) {
        if (num == -1) {
            i++; j = 0;
        } else {
            fc[i][j] = num;
            j++;
        }
    }
    fclen = i;
    int id = 0;
    i = 0; j = 0;
    double xx, yy, zz;
    double umin = 100000.0, umax = -10000.0;
    double vmin = 100000.0, vmax = -10000.0;
    double no[3];
    while (input_pt >> xx >> yy >> zz >> no[0] >> no[1] >> no[2]) {
        fcPoints[id] = Point(id, xx, yy, zz);
        umin = min(umin, fcPoints[id].u);
        vmin = min(vmin, fcPoints[id].v);
        umax = max(umax, fcPoints[id].u);
        vmax = max(vmax, fcPoints[id].v);
        //printf("%.3f %.3f %.3f\n", xx, yy, zz);
        id++;
    } 
    id = 0;
    sc =  (vmax - vmin) / (umax - umin);
    //printf("sc %f\n", sc*sc); 
    for (int i = 0; i < fclen; ++i) {
        xx = yy = zz = 0;
        for (int j = 0; j < 3; ++j) {
            xx += fcPoints[fc[i][j]].x;
            yy += fcPoints[fc[i][j]].y;
            zz += fcPoints[fc[i][j]].z;
        }
        xx /= 3.0; yy /= 3.0; zz /= 3.0;
        centerPoint[id] = Point(id, xx, yy, zz);
        id++;
    }
    cntFcPoint = id;
    id = 0;
    while (ref_pt >> xx >> yy >> zz >> no[0] >> no[1] >> no[2]) {
        Point rf;
        rf = Point(id, xx, yy, zz);
        int minID = 0;
        double minLen = 100000.0;
        for (int i = 0; i < cntFcPoint; ++i) {
            double tlen = rf.dis(centerPoint[i], sc);
            //printf("%f\n", tlen);
            if (tlen < minLen) {
                minLen = tlen;
                minID = i;
            }
        }
        id++;
        //if (id % 1000 == 0) printf("id %d\n", id);
        resampled << centerPoint[minID].x 
                  << " " << centerPoint[minID].y
                  << " " << centerPoint[minID].z << endl;
    }
    input_fc.close();
    input_pt.close();
    ref_pt.close();
    resampled.close();
}


int main() {
    struct dirent *ptr;    
    DIR *dir;
    string PATH = "./file";
    dir = opendir(PATH.c_str()); 
    vector<string> files;
    // cout << "文件列表: "<< endl;
    while((ptr=readdir(dir))!=NULL) {
        //跳过'.'和'..'两个目录
        if(ptr->d_name[0] == '.')
            continue;
        string s = PATH + "/" + ptr->d_name;
        if (s[s.length()-1] == 'z') {
            string tmp = ptr->d_name;
            //if (tmp > "M0015_SA03WH_F3D.xyz") {
            files.push_back(s);
            //cout << ptr->d_name << endl;
            //}
        }
    }
    //M0024_SU04AE_F3D
    
    string fcReferPt = "faceReference.xyz";
    printf("In Total %lu Faces\n", files.size());
    for (int i = 0; i < files.size(); ++i) {
        printf("Now Resample Face No.%d ...\n", i+1);
        // turn .xyz into .fc
        string fc = files[i].substr(0, files[i].length()-3);
        string bnd = fc + ".bnd";
        fc = fc + "fc";
        resample(fc, files[i], fcReferPt);
        printf("Done\n");
    }
    closedir(dir);

    return 0;
}
