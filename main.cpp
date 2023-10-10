#include <bits/stdc++.h>

using namespace std;

#define endl "\n"

template<typename T>
class Matrix
{
protected:
    int n, m;
    vector<vector<T>> grid;
public:
    Matrix()
    {
        n = 0;
        m = 0;
    }
    Matrix(int n0, int m0)
    {
        n = n0;
        m = m0;

        grid.assign(n, vector<T>(m, 0));
    }

    friend istream& operator>>(istream& is, Matrix<T> &A);
    friend ostream& operator<<(ostream& os, Matrix<T> &A);

    int getN()
    {
        return n;
    }

    int getM()
    {
        return m;
    }

    void setAtIndex(int i, int j, T v)
    {
        grid[i][j] = v;
    }

    T getAtIndex(int i, int j)
    {
        return grid[i][j];
    }

    vector<vector<T>>* getGrid()
    {
        return &grid;
    }


    bool operator==(Matrix &A)
    {
        if(n != A.getN() || m != A.getM())
        {
            return false;
        }

        vector<vector<T>> *temp = A.getGrid();
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                if(grid[i][j] != (*temp)[i][j])
                    return false;
            }
        }

        return true;
    }

    void operator=(Matrix &A)
    {
        if(n != A.getN() || m != A.getM())
        {
            cout << "Error: the dimensional problem occurred" << endl;
            return;
        }

        vector<vector<T>> *temp = A.getGrid();
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                grid[i][j] = (*temp)[i][j];
            }
        }
    }

    Matrix operator+(Matrix &A)
    {
        if(n != A.getN() || m != A.getM())
        {
            cout << "Error: the dimensional problem occurred" << endl;
            Matrix err(0, 0);
            return err;
        }

        Matrix D(n, m);

        vector<vector<T>> *tempA = A.getGrid();
        vector<vector<T>> *tempD = D.getGrid();
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                (*tempD)[i][j] = grid[i][j] + (*tempA)[i][j];
            }
        }

        return D;
    }

    Matrix operator-(Matrix &A)
    {
        if(n != A.getN() || m != A.getM())
        {
            cout << "Error: the dimensional problem occurred" << endl;
            Matrix err(0, 0);
            return err;
        }

        Matrix E(n, m);

        vector<vector<T>> *tempA = A.getGrid();
        vector<vector<T>> *tempE = E.getGrid();
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                (*tempE)[i][j] = grid[i][j] - (*tempA)[i][j];
            }
        }

        return E;
    }

    Matrix operator*(Matrix &A)
    {
        if(m != A.getN())
        {
            cout << "Error: the dimensional problem occurred" << endl;
            Matrix err(0, 0);
            return err;
        }

        Matrix F(n, A.getM());

        vector<vector<T>> *tempA = A.getGrid();
        vector<vector<T>> *tempF = F.getGrid();

        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<A.getM(); j++)
            {
                int counter = 0;
                while(counter < m)
                {
                    (*tempF)[i][j] += grid[i][counter]*(*tempA)[counter][j];
                    counter++;
                }
            }
        }

        return F;
    }

    Matrix operator-()
    {
        Matrix temp = *this;
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                temp.setAtIndex(i, j, -grid[i][j]);
            }
        }

        return temp;
    }

    Matrix transpose()
    {
        Matrix G(m, n);
        vector<vector<T>> *temp = G.getGrid();

        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                (*temp)[j][i] = grid[i][j];
            }
        }

        return G;
    }

    Matrix* augmentedMatrix()
    {
        Matrix* aug = new Matrix(n, 2*m);

        vector<vector<T>> *temp = aug->getGrid();

        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                (*temp)[i][j] = grid[i][j];
            }
        }

        for(int i = 0; i<n; i++)
        {
            (*temp)[i][i+m] = 1;
        }

        return aug;
    }
};

template<typename T>
class SquareMatrix: public Matrix<T>
{
public:
    SquareMatrix()
    {
        Matrix<T>::n = 0;
        Matrix<T>::m = 0;
    }

    SquareMatrix(int n0)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));
    }
};

template<typename T>
class IdentityMatrix: public SquareMatrix<T>
{
public:
    IdentityMatrix()
    {
        Matrix<T>::n = 0;
        Matrix<T>::m = 0;
    }

    IdentityMatrix(int n0)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int i = 0; i<Matrix<T>::n; i++)
        {
            Matrix<T>::grid[i][i] = 1;
        }
    }
};

template<typename T>
class EliminationMatrix: public SquareMatrix<T>
{
public:
    EliminationMatrix()
    {
        Matrix<T>::n = 0;
        Matrix<T>::m = 0;
    }

    EliminationMatrix(int n0)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int i = 0; i<Matrix<T>::n; i++)
        {
            Matrix<T>::grid[i][i] = 1;
        }
    }

    EliminationMatrix(int n0, Matrix<T>* M, int i, int j)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int k = 0; k<Matrix<T>::n; k++)
        {
            Matrix<T>::grid[k][k] = 1;
        }

        vector<vector<T>>* temp = M->getGrid();
        double c = (*temp)[i-1][j-1]/(*temp)[j-1][j-1];

        Matrix<T>::grid[i-1][j-1] = -c;
    }

    void eliminate(Matrix<T>* M, int i, int j)
    {
        vector<vector<T>>* temp = M->getGrid();
        double c = (*temp)[i-1][j-1]/(*temp)[j-1][j-1];

        Matrix<T>::grid[i-1][j-1] = -c;
    }
};

template<typename T>
class PermutationMatrix: public SquareMatrix<T>
{
public:
    PermutationMatrix()
    {
        Matrix<T>::n = 0;
        Matrix<T>::m = 0;
    }

    PermutationMatrix(int n0)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int i = 0; i<Matrix<T>::n; i++)
        {
            Matrix<T>::grid[i][i] = 1;
        }
    }

    PermutationMatrix(int n0, Matrix<T>* M, int i, int j)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int k = 0; k<Matrix<T>::n; k++)
        {
            Matrix<T>::grid[k][k] = 1;
        }

        swap(Matrix<T>::grid[i-1], Matrix<T>::grid[j-1]);
    }

    void permute(Matrix<T>* M, int i, int j)
    {
        swap(Matrix<T>::grid[i-1], Matrix<T>::grid[j-1]);
    }
};

istream& operator>>(istream &is, Matrix<double> *A)
{
    vector<vector<double>> *temp = A->getGrid();
    for(int i = 0; i<A->getN(); i++)
    {
        for(int j = 0; j<A->getM(); j++)
        {
            is >> (*temp)[i][j];
        }
    }

    return is;
}

ostream& operator<<(ostream &os, Matrix<double> &A)
{
    vector<vector<double>> *temp = A.getGrid();
    for(int i = 0; i<A.getN(); i++)
    {
        for(int j = 0; j<A.getM(); j++)
        {
            os << fixed << setprecision(6) << (*temp)[i][j];
            if(j <A.getM()-1)
            {
                cout << ' ';
            }
        }
        cout << endl;
    }

    return os;
}

Matrix<double> calculateInverse(Matrix<double> A, int n)
{
    Matrix<double>* aug = A.augmentedMatrix();

    int step = 1;
    for(int j = 1; j<=n-1; j++)
    {
        int r1 = j, r2 = j;
        for(int i = j+1; i<=n; i++)
        {
            double temp1 = (*(A.getGrid()))[r2-1][j-1];
            double temp2 = (*(A.getGrid()))[i-1][j-1];

            if(fabs(temp2) > fabs(temp1))
            {
                r2 = i;
            }
        }

        if(r1 != r2)
        {
            Matrix<double>* P = new PermutationMatrix<double>(n, &A, r1, r2);
            Matrix<double> temp = (*P)*(A);
            A = temp;

            Matrix<double> temp2 = (*P)*(*aug);
            *aug = temp2;

            step++;
        }

        for(int i = j+1; i<=n; i++)
        {
            Matrix<double>* E = new EliminationMatrix<double>(n, &A, i, j);
            Matrix<double> temp = (*E)*(A);

            if(A == temp)
            {
                continue;
            }

            A = temp;

            Matrix<double> temp2 = (*E)*(*aug);
            *aug = temp2;

            step++;
        }
    }


    for(int j = n; j>=2; j--)
    {
        for(int i = j-1; i >= 1; i--)
        {
            Matrix<double>* E = new EliminationMatrix<double>(n, &A, i, j);

            Matrix<double> temp = (*E)*(A);

            if(A == temp)
            {
                continue;
            }

            A = temp;

            Matrix<double> temp2 = (*E)*(*aug);
            *aug = temp2;

            step++;
        }
    }

    for(int i = 1; i<=n; i++)
    {
        for(int j = i+1; j<=2*n; j++)
        {
            (*(aug->getGrid()))[i-1][j-1] /= (*(aug->getGrid()))[i-1][i-1];
        }

        (*(aug->getGrid()))[i-1][i-1] = 1;
    }

    for(int i = 1; i<=n; i++)
    {
        for(int j = 1; j<=n; j++)
        {
            (*(A.getGrid()))[i-1][j-1] = (*(aug->getGrid()))[i-1][j-1+n];
        }
    }

    return A;
}


bool solution_found(vector<double> &z_minus_c, int n, double precision)
{
    for(int i = 0; i<n; i++)
    {
        if(z_minus_c[i] + precision < 0)
            return false;
    }

    return true;
}

bool unbounded_solution(Matrix<double> &curr)
{
    for(int i = 0; i<curr.getN(); i++)
    {
        if(curr.getAtIndex(i, 0) > 0)
            return false;
    }

    return true;
}

void simplexMethod(Matrix<double>* C, Matrix<double>* A, Matrix<double>* b, double precision, string &purpose)
{
    int m = A->getN(); //number of constraints
    int n = A->getM() - m; //number of coefficients in the original objective function
    //the size of C is n+m
    //The size of b is m


    // Minimum case
    if(purpose == "minimum")
    {
        for(int i = 0; i<m; i++)
        {
            C->setAtIndex(0, i, -(C->getAtIndex(0, i)));
        }
    }
    else if(purpose != "maximum")
    {
        cout << "You have to chose either minimum or maximum!" << endl;
        return;
    }

    vector<int> basic_vars, non_basic_vars;
    for(int i = 0; i<n+m; i++)
    {
        if(i<n)
            non_basic_vars.push_back(i);
        else
            basic_vars.push_back(i);
    }

    // Step 0: Construct the initial basic feasible solution
    Matrix<double> P = A->transpose(); //(n+m, m)

    Matrix<double> *B = new IdentityMatrix<double>(m);

    Matrix<double> *C_b = new Matrix<double>(1, m);
    for(int i = 0; i<m; i++)
    {
        C_b->setAtIndex(0, i, C->getAtIndex(0, C->getM()-m+i));
    }

    int nb_iteration = 0;
    cout << "Iteration " << nb_iteration << ':' << endl;
    cout << "B:" << endl;
    cout << *B;
    cout << "C_b:" << endl;
    cout << *C_b << endl;
    nb_iteration++;

    while(true)
    {
        vector<double> z_minus_c(n, 0);
        // Step 1: Compute the value of B^(-1) for the basis B using the inversion method
        Matrix<double> B_inverse = calculateInverse(*B, m);

        Matrix<double> x_b = B_inverse * (*b); //values of basic variables

        // Step 2: Calculate z_i - c_i for each i from 0 to n-1
        for(int i = 0; i<n; i++)
        {
            Matrix<double> *P_i = new Matrix<double>(m, 1);
            for(int j = 0; j<m; j++)
            {
                P_i->setAtIndex(j, 0, P.getAtIndex(non_basic_vars[i], j));
            }

            Matrix<double> temp = ((*C_b) * B_inverse * (*P_i));
            z_minus_c[i] = temp.getAtIndex(0, 0) - C->getAtIndex(0, non_basic_vars[i]);
        }

        if(solution_found(z_minus_c, n, precision))
        {
            Matrix<double> temp = (*C_b) * x_b;
            double z = temp.getAtIndex(0, 0);

            cout << "Optimal solution found!" << endl;
            cout << "The vector of decision variables:" << endl;
            cout << x_b << endl;
            cout << "The " << purpose << " value of the objective function:" << endl;
            cout << z << endl;

            return;
        }
        else
        {
            // Step 3: Find P_entering with the smallest z_i - c_i
            //(for maximization) or the largest (for minimization).

            Matrix<double> *P_entering = new Matrix<double>(m, 1);
            int entering_i = 0;
            for(int i = 0; i<n; i++)
            {
                if(z_minus_c[i] < z_minus_c[entering_i])
                    entering_i = i;
            }

            for(int j = 0; j<m; j++)
            {
                P_entering->setAtIndex(j, 0, P.getAtIndex(non_basic_vars[entering_i], j));
            }

            // Step 4: Compute B^(-1) * P_entering
            Matrix<double> curr = B_inverse * (*P_entering);
            if(unbounded_solution(curr))
            {
                cout << "Method inapplicable!" << endl;
                return;
            }
            else
            {
                // Step 5: find the leaving vector
                Matrix<double> *P_leaving = new Matrix<double>(m, 1);
                int leaving_i = 0;
                for(int j = 0; j<m; j++)
                {
                    if( x_b.getAtIndex(j, 0)/curr.getAtIndex(j, 0) < x_b.getAtIndex(leaving_i, 0)/curr.getAtIndex(leaving_i, 0) )
                        leaving_i = j;
                }

                for(int j = 0; j<m; j++)
                {
                    P_leaving->setAtIndex(j, 0, P.getAtIndex(basic_vars[leaving_i], j));
                }

                // Update B
                for(int i = 0; i<m; i++)
                {
                    B->setAtIndex(i, leaving_i, P_entering->getAtIndex(i, 0));
                }

                // Update C_b

                C_b->setAtIndex(0, leaving_i, C->getAtIndex(0, non_basic_vars[entering_i]));

                // Update basic variables
                int temp = non_basic_vars[entering_i];
                non_basic_vars[entering_i] = basic_vars[leaving_i];
                basic_vars[leaving_i] = temp;

                cout << "Iteration " << nb_iteration << ':' << endl;
                cout << "B:" << endl;
                cout << *B;
                cout << "C_b:" << endl;
                cout << *C_b << endl;

                cout << "Basic variables:" << endl;
                for(auto elt: basic_vars)
                    cout << elt+1 << ' ';
                cout << endl << endl;
                cout << "Non Basic variables:" << endl;
                for(auto elt: non_basic_vars)
                    cout << elt+1 << ' ';
                cout << endl << endl;

            }
        }
        nb_iteration++;
    }
}

int main()
{
    cout << "NOTE: The problem must be in standard form! If it is not, rewrite the problem in standard form." << endl;
    cout << "-----------------------------------------------------------------------------------------------" << endl;
    int n;
    cout << "Type the number of variables in the objective function:" << endl;
    cin >> n;

    int m;
    cout << "Type the number of constraint functions:" << endl;
    cin >> m;

    Matrix<double> *C = new Matrix<double>(1, n+m);
    cout << "Type the " << n+m << " coefficients of the augmented objective function, seperated by spaces:" << endl;
    cin >> C;

    Matrix<double> *A = new Matrix<double>(m, n+m);
    cout << "Type the coefficients of the constraint function. Each of the " << m << " lines must contain " << n+m << " coefficients, seperated by spaces:" << endl;
    cin >> A;

    Matrix<double> *b = new Matrix<double>(m, 1);
    cout << "Type the right hand side of the constraints containing " << m << " numbers, seperated by spaces:" << endl;
    cin >> b;

    double precision;
    cout << "Type the approximation accuracy:" << endl;
    cin >> precision;

    string purpose;
    cout << "Type maximum or minimum:" << endl;
    cin >> purpose;

    simplexMethod(C, A, b, precision, purpose);

    return 0;
}
