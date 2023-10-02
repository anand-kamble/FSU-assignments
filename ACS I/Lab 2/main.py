import numpy as np


class PageRank():
    def __init__(self, n, debug=False) -> None:
        self.n = n
        self.A = None
        self.d = 0.85
        self.I = None
        self.M = None
        self.R = None

        self.debug = debug
        pass

    def validateMatrix(self):
        for row in self.A:
            if not 1 in row:
                self.createMatrix()
                self.validateMatrix()
        return self

    def createMatrix(self):
        self.A = np.random.randint(2, size=(self.n, self.n))
        for i in range(self.n):
            self.A[i][i] = 0
        self.I = np.identity(self.n)
        self.M = self.A/self.A.sum(axis=0)
        self.R = np.random.randint(0, 2, (self.n))
        print(
            f"Created R matrix of size {self.n}x{self.n}\n", self.R) if self.debug else 0
        return self

    def getRanksMethod_1(self):
        self.createMatrix().validateMatrix()
        self.R = np.linalg.solve(
            (self.I - self.d*self.M), (((1 - self.d) / self.n) * np.ones(self.n)))
        return self.R.copy()

    def getRanksMethod_2(self):
        error = 1
        tolerance = 1e-10
        iteration = 1
        while (error > tolerance):
            E = np.ones((self.n, self.n))
            M_bar = self.d*self.M + ((1 - self.d) / self.n)*E

            R_new = self.normalize(M_bar @ self.R)
            print("R_new is \n", R_new) if self.debug else 0

            Lambda = (R_new.transpose() @ M_bar @ R_new) / \
                (R_new.transpose() @ R_new)

            error = np.abs(np.linalg.norm(M_bar @ self.R - Lambda * self.R))

            self.R = R_new
            print(
                f"Error at iteration {iteration} = {error}") if self.debug else 0
            iteration = iteration + 1

        return self.R.copy()

    def normalize(self, matrix: np.ndarray):
        return (matrix/np.linalg.norm(matrix)).copy()


rank = PageRank(5, debug=True)

print("Solution Method 1 \n", rank.getRanksMethod_1())
print("Solution Method 2 \n", rank.getRanksMethod_2())
