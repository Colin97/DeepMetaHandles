#include <igl/biharmonic_coordinates.h>
#include <igl/matrix_to_list.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/remove_unreferenced.h>
#include <igl/slice.h>
#include <iostream>

struct Mesh
{
    Eigen::MatrixXd V, U;
    Eigen::MatrixXi T, F;
} low, high, surface;

Eigen::MatrixXd W;

int main(int argc, char *argv[])
{
    using namespace Eigen;
    using namespace std;
    using namespace igl;

    if (!readMESH(argv[1], high.V, high.T, high.F)) //tet mesh
    {
        cout << "failed to load mesh" << endl;
    }

    if (!readOBJ(argv[2], surface.V, surface.F)) // surface mesh
    {
        cout << "failed to load mesh" << endl;
    }

    if (!readOBJ(argv[3], low.V, low.F)) // key points
    {
        cout << "failed to load mesh" << endl;
    }

    Eigen::VectorXi bb;
    {
        Eigen::VectorXi J = Eigen::VectorXi::LinSpaced(high.V.rows(), 0, high.V.rows() - 1);
        Eigen::VectorXd sqrD;
        Eigen::MatrixXd _2;
        igl::point_mesh_squared_distance(surface.V, high.V, J, sqrD, bb, _2);
        assert(sqrD.minCoeff() < 1e-7 && "surface.V should exist in high.V");
    }
    for_each(surface.F.data(), surface.F.data() + surface.F.size(), [&bb](int &a) { a = bb(a); });
    high.F = surface.F;

    {
        Eigen::VectorXi b;
        {
            Eigen::VectorXi J = Eigen::VectorXi::LinSpaced(high.V.rows(), 0, high.V.rows() - 1);
            Eigen::VectorXd sqrD;
            Eigen::MatrixXd _2;
            igl::point_mesh_squared_distance(low.V, high.V, J, sqrD, b, _2);
            assert(sqrD.minCoeff() < 1e-7 && "low.V should exist in high.V");
        }

        igl::slice(high.V, b, 1, low.V);

        std::vector<std::vector<int>> S;
        igl::matrix_to_list(b, S);
        cout << "Computing weights for " << b.size() << " handles at " << high.V.rows() << " vertices..." << endl;
        const int k = 2;
        
        igl::biharmonic_coordinates(high.V, high.T, S, k, W);
    
        VectorXi I, J;
        //cout << high.V.rows() << endl;
        igl::remove_unreferenced(high.V.rows(), high.F, I, J);
        for_each(high.F.data(), high.F.data() + high.F.size(), [&I](int &a) { a = I(a); });
        for_each(b.data(), b.data() + b.size(), [&I](int &a) { a = I(a); });
        igl::slice(MatrixXd(high.V), J, 1, high.V);
        igl::slice(MatrixXd(W), J, 1, W);
    }

    freopen(argv[4], "w", stdout); // weights_mesh
    printf("%d %d\n", W.rows(), W.cols());
    for (int i = 0; i < W.rows(); i++)
    {
        for (int j = 0; j < W.cols(); j++)
            printf("%f ", W(i, j));
        printf("\n");
    }
}
