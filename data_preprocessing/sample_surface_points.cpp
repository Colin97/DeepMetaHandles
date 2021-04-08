#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <cmath>

struct Mesh
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
} surface, output;

struct Point
{
    double x, y, z;
    Point(){};
    Point(double _x, double _y, double _z)
    {
        x = _x;
        y = _y;
        z = _z;
    };
    Point operator-(const Point &v) const
    {
        return Point(x - v.x, y - v.y, z - v.z);
    };

    Point operator+(const Point &v) const
    {
        return Point(x + v.x, y + v.y, z + v.z);
    };

    Point operator*(const double t) const
    {
        return Point(x * t, y * t, z * t);
    }

    double length()
    {
        return sqrt(x * x + y * y + z * z);
    }

    void normalize()
    {
        double l = length();
        x /= l;
        y /= l;
        z /= l;
        return;
    }

    float dot(const Point &v) const
    {
        return x * v.x + y * v.y + z * v.z;
    };

    Point cross(const Point &v) const
    {
        return Point(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x);
    };

    void print()
    {
        printf("%lf %lf %lf\n", x, y, z);
    }
};

Point randomPointTriangle(Point a, Point b, Point c)
{
    double r1 = (double)rand() / RAND_MAX;
    double r2 = (double)rand() / RAND_MAX;
    double r1sqr = std::sqrt(r1);
    double OneMinR1Sqr = (1 - r1sqr);
    double OneMinR2 = (1 - r2);
    a = a * OneMinR1Sqr;
    b = b * OneMinR2;
    return (c * r2 + b) * r1sqr + a;
}


Eigen::MatrixXd weights(500000, 500);

int main(int argc, char *argv[]) {

    freopen(argv[1], "r", stdin); // w_mesh
    int n, m, n_keypoints;
    scanf("%d %d", &n, &n_keypoints);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n_keypoints; j++) {
            double t;
            scanf("%lf", &t);
            weights(i, j) = t;
        }

    if (!igl::readOBJ(argv[2], surface.V, surface.F)) // surface mesh
    {
        std::cout << "failed to load mesh" << std::endl;
    }
    assert(n == surface.V.rows());
    m = surface.F.rows();

    std::vector<Point> mesh_vertices;
    for (int i = 0; i < n; i++)
    {
        mesh_vertices.push_back(Point(surface.V(i, 0), surface.V(i, 1), surface.V(i, 2)));
    }

    float surface_area = 0;
    for (int i = 0; i < m; i++)
    {
        Point A = mesh_vertices[surface.F(i, 0)];
        Point B = mesh_vertices[surface.F(i, 1)];
        Point C = mesh_vertices[surface.F(i, 2)];
        surface_area += (B - A).cross(C - A).length();
    }
    //printf("%f\n", surface_area);

    const int n_sampled_points = 4096;
    int cnt = 0;
    int tt = m / n_sampled_points + 2;
    Eigen::MatrixXd sampled_points(n_sampled_points, 3);
    Eigen::MatrixXd null_set;
    
    freopen(argv[3], "w", stdout); // w_mesh_4096
    printf("%d %d\n", n_sampled_points, n_keypoints);
    for (int i = 0; i < m; i++)
    {
        int a = surface.F(i, 0);
        int b = surface.F(i, 1);
        int c = surface.F(i, 2);
        Point A = mesh_vertices[a];
        Point B = mesh_vertices[b];
        Point C = mesh_vertices[c];
        float area = (B - A).cross(C - A).length();
        int t = std::max(1.0f, n_sampled_points * tt * (area / surface_area));
        if (i == m - 1)
            t = n_sampled_points * tt - t;
        if (cnt + t > n_sampled_points * tt)
            t = n_sampled_points * tt - cnt;
        for (int j = 0; j < t; j++) {
            if (cnt % tt == 0) {
                Point p = randomPointTriangle(A, B, C);
                float wa = (B - p).cross(C - p).length() / area;
                float wb = (A - p).cross(C - p).length() / area;
                float wc = (A - p).cross(B - p).length() / area;
                for (int k = 0; k < n_keypoints; k++) {
                    float w = 0;
                    w += weights(a, k) * wa;
                    w += weights(b, k) * wb;
                    w += weights(c, k) * wc;
                    printf("%f ", w);
                }
                printf("\n");
                sampled_points(cnt / tt, 0) = p.x;
                sampled_points(cnt / tt, 1) = p.y;
                sampled_points(cnt / tt, 2) = p.z;
            }
            cnt++;
        }
    }

    igl::writeOBJ(argv[4], sampled_points, null_set); // pc_4096
    return 0;
}
