#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <cmath>
#include <queue>
#include <igl/slice.h>

struct Mesh
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
} surface, output;

struct Edge{
  int u, v, n;
  float l;
};

std::vector<Edge> edges;
std::vector<int> head;
std::vector<float> dis;
std::vector<bool> is_key_point;
int n, m;

void add(int u, int v) {
  Edge e;
  e.u = u; e.v = v;
  float dx = surface.V(u, 0) - surface.V(v, 0);
  float dy = surface.V(u, 1) - surface.V(v, 1);
  float dz = surface.V(u, 2) - surface.V(v, 2);
  e.l = sqrt(dx * dx + dy * dy + dz * dz);
  e.n = head[u];
  head[u] = edges.size();
  edges.push_back(e);
}

void spfa(int st) {
  std::queue<int> Q;
  Q.push(st);
  std::vector<int> inq;
  inq.resize(n, false);
  inq[st] = true;
  while (!Q.empty()) {
    int u = Q.front(); Q.pop(); 
    inq[u] = false;
    for (int i = head[u]; i != -1; i = edges[i].n) {
      int v = edges[i].v;
      if (!is_key_point[v] && dis[v] > dis[u] + edges[i].l) {
        dis[v] = dis[u] + edges[i].l;
        if (!inq[v]) {
          inq[v] = true;
          Q.push(v);
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  printf("%s\n", argv[1]);

  if(!igl::readOBJ(argv[1],surface.V,surface.F))
  {
    std::cout<<"failed to load mesh"<<std::endl;
  }
  n = surface.V.rows();
  m = surface.F.rows();
  
  head.resize(n, -1);
  dis.resize(n, 1e9);
  is_key_point.resize(n, false);
  for (int i = 0; i < m; i++) {
    int a = surface.F(i, 0), b = surface.F(i, 1), c = surface.F(i, 2);
    add(a, b); add(b,a); add(a,c); add(c,a); add(b,c); add(c,b);
  }
  //std::cout << edges.size() << std::endl;
  const int n_key_points = atoi(argv[3]);
  Eigen::VectorXi key_points(n_key_points);
  for (int i = 0; i < n_key_points; i++) {
    int u = -1;
    if (i == 0)
      u = 0;
    else {
      for (int j = 0; j < n; j++)
        if (is_key_point[j] == false && (u == -1 || dis[j] > dis[u]))
          u = j;
    }
    dis[u] = 0;
    is_key_point[u] = true;
    spfa(u);
    key_points[i] = u;
  }

  igl::slice(surface.V,key_points,1,output.V);
  igl::writeOBJ(argv[2],output.V,output.F);
  return 0;
}
