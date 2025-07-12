import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import linprog
from scipy.optimize import LinearConstraint
from scipy.spatial import ConvexHull 

from stl_constrained_zpc.scripts.reachability.contSet import contSet
from stl_constrained_zpc.scripts.reachability.Interval import Interval


class Zonotope():
    def __init__(self, *args):
        """
        Zonotope class constructor. Initializes a zonotope object. 

        Args:
            *args: 0 or 1 arguments. If no arguments are provided, an empty zonotope is created. If one argument is provided,
            it can be a center and generator matrix, a zonotope object, or an interval object. If two arguments are provided,
            they are the center and generator matrix of the zonotope. 
        """
        self.Z = np.array([])
        self.half_space = np.array([])
        self.cont_set = np.array([])

        if(len(args) == 0):
            self.cont_set = contSet()

        elif(len(args) == 1):
            if(len(np.shape(args[0])) == 1 or isinstance(args[0], Interval)):
                self.cont_set = contSet(1)
            else:
                self.cont_set = contSet(np.shape(args[0])[0])

            if(type(args[0]) == Zonotope):
                self.copy(args[0])
            
            elif(type(args[0]) == Interval):
                center = 0.5 * (args[0].inf + args[0].sup)
                G = np.diag(0.5 * (np.array(args[0].sup) - np.array(args[0].inf)).flatten())
                self.Z = np.hstack((center.reshape(-1, 1), G))
                self.half_space = np.array([])

            else:
                self.Z = np.copy(args[0])
                self.half_space = np.array([])

        elif(len(args) == 2):
            if(len(np.shape(args[0])) == 1 or len(np.shape(args[0])) == 0):
                self.cont_set = contSet(1)
            else:
                self.cont_set = contSet(np.shape(args[0])[0])
            
            self.Z = np.hstack([args[0], args[1]])
            self.half_space = np.array([])
        
    def center(self):
        """
        Get the center of the zonotope. 

        Returns:
            np.array: Center of the zonotope.
        """
        if(len(self.Z.shape) == 1):
            return np.array([self.Z[0]]).reshape(-1, 1)
        else:
            return self.Z[:, 0:1]

    def generators(self):
        """
        Get the generator matrix of the zonotope. 

        Returns:
            np.array: Generator matrix of the zonotope.
        """
        if(len(self.Z.shape) == 1):
            return np.array([self.Z[1]]).reshape(-1, 1)
        else:
            return self.Z[:, 1:]

    def copy(self, zon):
        """
        Copy the zonotope object. 

        Args:
            zon (object): Zonotope object to copy.
        """
        self.Z = np.copy(zon.Z) 
        self.half_space = np.copy(zon.half_space )
        self.cont_set = zon.cont_set
        
        return self

    def __add__(self, operand):
        """
        Add two zonotopes. 

        Args:
            operand (object): Zonotope object or np.array to add to the current zonotope.
        
        Returns:
            object: Zonotope object resulting from the addition.
        """
        Z = Zonotope(self.Z)

        if(type(operand) == Zonotope):
            Z.Z[:, 0:1] = Z.Z[:, 0:1] + operand.Z[:, 0:1]
            Z.Z = np.hstack((Z.Z, operand.Z[:, 1:]))
        
        elif(type(operand) == np.ndarray and (len(operand.shape) == 1 or operand.shape[1] == 1)):
            Z.Z[:, 0:1] = Z.Z[:, 0:1] + operand

        elif(type(operand) == np.ndarray):
            Z.Z = Z.Z + Zonotope(operand)

        else:
            raise Exception("Invalid argument for addidtion")

        return Z 

    def __sub__(self, operand):
        """
        Subtract a scalar or a matrix from a zonotope.

        Args:
            operand (float, int, np.array): Scalar or matrix to subtract from the zonotope.
        """
        Z = Zonotope(self.Z)
        if(type(operand) == Zonotope):
            raise Exception("Zonotopes subtraction is not supported when both operands are zonotopes")
        elif(type(operand) == np.ndarray and (len(operand.shape) == 1 or operand.shape[1] == 1)):
            Z.Z[:, 0:1] = self.Z[:, 0:1] - operand

        return Z

    def __mul__(self, operand):
        """
        Multiply a zonotope by a scalar or a matrix.

        Args:
            operand (float, int, np.array): Scalar or matrix to multiply the zonotope by.
        
        Returns:
            object: Zonotope object resulting from the multiplication.
        """
        if(isinstance(operand, float) or isinstance(operand, int)):
            Z = Zonotope(self.Z)
            Z.Z = Z.Z * operand
            
            return Z
        
        else:
            Z = Zonotope(self.Z)
            Z.Z = np.dot(operand, Z.Z.reshape((operand.shape[1], -1))) 

            return Z
    
    __rmul__ = __mul__   # commutative operation

    def __and__(self, operand):
        """
        Perform the intersection of two zonotopes. 

        Args:
            operand (object): Zonotope object to intersect with.
        
        Returns:
            object: Zonotope object resulting from the intersection.
        """
        if(isinstance(operand, Zonotope)):
            c, G, A, b = self.intersect_zonotopes(operand)
            Z = Zonotope(c, G)

            return Z
        
        else:
            raise Exception("Intersection is only implemented for zonotopes")

    def reduce(self, option, *args):
        """
        Reduce the zonotope using a specific reduction method. 

        Args:
            option (str): Reduction method to use.
            *args: Additional arguments for the reduction method. If no arguments are provided, the default reduction method is used. 
            If one argument is provided, it is the order of the reduction. 

        Returns:
            object: Reduced zonotope object.
        """
        if(len(args) == 0):
            order = 1
            filterLength = []

        elif(len(args) == 1):
            order = args[0]
            filterLength = []

        elif(len(args) == 2):
            order = args[0]
            filterLength = args[1]
        
        elif(len(args) == 3):
            order = args[0]
            filterLength = args[1]
            method = args[2]

        elif(len(args) == 4):
            order = args[0]
            filterLength = args[1]
            method = args[2]
            alg = args[3]

        if(option == "girard"):
            Zred = self.reduce_girard(order)
        else:
            raise Exception("Other Reduction methods are not implemented yet")

        return Zred
        
    def reduce_girard(self, order):
        """
        Reduce the zonotope using the Girard method.

        Args:
            order (int): Order of the reduction.

        Returns:
            object: Reduced zonotope object.
        """
        Zred = Zonotope(self.Z)

        center, Gunred, Gred = Zred.picked_generators(order)
        if(Gred.size == 0):
            Zred.Z = np.hstack((center, Gunred))
        else:
            d = np.sum(np.abs(Gred), axis=1)
            Gbox = np.diag(d)
            center = center.reshape((center.shape[0], -1))
            Gunred = Gunred.reshape((center.shape[0], -1))
            Gbox = Gbox.reshape((center.shape[0], -1))
            Zred.Z = np.hstack((center, Gunred, Gbox))#np.array([[center], [Gunred], [Gbox]])
        
        return Zred

    def picked_generators(self, order):
        """
        Pick the generators of the zonotope based on a specific order.

        Args:
            order (int): Order of the reduction.

        Returns:
            np.array: Center of the zonotope.
        """
        Z = Zonotope(self.Z)
        c = Z.center()
        G = Z.generators()

        Gunred = np.array([])
        Gred = np.array([])

        if(np.sum(G.shape) != 0):
            G = self.nonzero_filter(G)
            d, nr_of_gens = G.shape
            if(nr_of_gens > d * order):
                h = np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 0, G)  - \
                    np.apply_along_axis(lambda row:np.linalg.norm(row,ord=np.inf), 0, G)
                n_unreduced = np.floor(d * (order - 1))
                n_reduced = int(nr_of_gens - n_unreduced)
                idx = np.argpartition(h, n_reduced - 1)
                Gred   = G[:, idx[: n_reduced]]
                Gunred = G[:, idx[n_reduced: ]]

            else:
                Gunred = G 
        
        return c, Gunred, Gred

    def nonzero_filter(self, generators):
        """
        Nonzero Filter Function. 

        Args:
            generators (np.array): Generator matrix of the zonotope.
        """
        idx = np.argwhere(np.all(generators[..., :] == 0, axis=0))
        
        return np.delete(generators, idx, axis=1)

    def cart_prod(self, other):
        """
        Cart Product Function. IMPORTANT NOTE: THIS function doesn't take into account order. It's somewhat messed up.
        However, it works fine with the current implementation of reachability. The part that needs modification is the 
        numpy.ndarray or list part. That is the concatenation of the array or the list should be reversed as it depends on the 
        order of multiplication

        Args:
            other (object): Zonotope object or np.array to perform the cartesian product with.

        Returns:
            object: Zonotope object resulting from the cartesian product.
        """
        if(isinstance(other, Zonotope)):
            center = np.vstack((self.center(), other.center()))
            G = block_diag(self.generators(), other.generators())
            Z = Zonotope(center, G)
            
            return Z
        
        elif(isinstance(other, np.ndarray) or isinstance(other, list)):
            other = np.array(other)
            center = np.vstack(( other, self.center()))
            G = np.vstack((np.zeros((other.shape[0], self.generators().shape[1])), self.generators()))
            result = Zonotope(center, G)
            
            return result
        
        else:
            raise Exception("cart products are only implemented if the two arguments are zonotopes")

    def __str__(self):
        """
        String representation of the zonotope object.

        Returns:
            str: String representation of the zonotope object.
        """
        S = "id: {} dimension: {} \n Z: \n {}".format(self.cont_set._id, self.cont_set._dimension, self.Z)

        return S 

    def project(self, proj):
        """
        Project the zonotope onto a specific dimension.

        Args:
            proj (list): List of dimensions to project onto.
        """
        Z = Zonotope(self.Z)
        Z.Z = Z.Z[proj, :]

        return Z 

    def polygon(self):
        """
        Generate the polygon of the zonotope. 

        Returns:
            np.array: Polygon of the zonotope.
        """
        c = self.center()
        G = self.generators()

        n = G.shape[1]

        xmax = np.sum(np.abs(G[0, :]))
        ymax = np.sum(np.abs(G[1, :]))

        Gnorm = np.copy(G)
        Gnorm[:, np.where(G[1, :] < 0)] = G[:, np.where(G[1, :] < 0)] * -1

        angles = np.arctan2(Gnorm[1, :], Gnorm[0, :])

        angles[np.where(angles < 0)] = angles[np.where(angles < 0)] + 2 * np.pi 

        IX = np.argsort(angles)
        p = np.zeros((2, n + 1))

        for i in range(n):
            p[:, i + 1] = p[:, i] + 2 * Gnorm[:, IX[i]]

        p[0, :] = p[0, :] + xmax - max(p[0, :])
        p[1, :] = p[1, :] - ymax

        p = np.vstack((np.hstack((p[0, :], p[0, -1] + p[0, 0] - p[0, 1:])),
                       np.hstack((p[1, :], p[1, -1] + p[1, 0] - p[1, 1:]))))
        
        p[0, :] = c[0] + p[0, :]
        p[1, :] = c[1] + p[1, :]

        return p

    def to_interval(self):
        """
        Convert the zonotope to an interval.

        Returns:
            object: Interval object.
        """
        result = Zonotope(self.Z)

        c = result.center()
        delta = np.sum(np.abs(result.Z), axis=1).reshape((-1, 1)) - np.abs(c)

        left_limit  = c - delta
        right_limit = c + delta 

        I = Interval(left_limit, right_limit)

        return I

    def plot(self, ax=None, color=None, alpha=0., label=None, *args):
        """
        Plot the zonotope. 

        Args:
            plot_label (bool): If True, adds a label to the plot.
            angle (float): Angle to rotate the zonotope.
            *args: Additional arguments for the plot method. If no arguments are provided, the default plot method is used. 
            If two arguments are provided, they are the dimensions to project onto. If three arguments are provided, the 
            second argument is the dimensions to project onto and the third argument is the line specification. If four arguments 
            are provided, the second argument is the dimensions to project onto, the third argument is the line specification,
            and the fourth argument is the filled flag.

        Returns:
            object: Zonotope object resulting from the multiplication.
        """
        # Check the dimensions of the zonotope
        dim = self.Z.shape[0]

        if dim >= 2:
            V = self.polygon()
            linespec = "k-"

            if V.shape[0] >= 2:
                xs = V[0, 1:]
                ys = V[1, 1:]
                if color is None:
                    colors = ["y"]
                else:
                    colors = [color]

            if ax is None:
                # Assign label to the filled area instead of the line
                plt.fill(xs, ys, np.random.choice(colors), alpha=alpha, label=label)
                plt.plot(V[0, :], V[1, :], linespec, alpha=min(alpha * 3, 1))  # Keep line separate

            else:
                ax.fill(xs, ys, np.random.choice(colors), alpha=alpha, label=label)  # Assign label here
                ax.plot(V[0, :], V[1, :], linespec, alpha=min(alpha * 3, 1))  # Keep line separate

        return ax

    
    def get_1D_center_generator(self):
        """
        Get the center and generator of the zonotope in 1D.

        Returns:
            np.array: Center of the zonotope.
            np.array: Generator of the zonotope.
        """
        c = self.center()[0]

        if len(self.generators().shape) > 1:
            G = self.generators()[0]
        else:
            G = self.generators()

        # Remove all zero generators if they exist
        if len(G) > 1:
            G = G[G != 0]

        return c, G

    def plot_1D(self, ax=None, c_history=None, G_history=None, color="blue", alpha=0.5, update=False, center=False, label=None, *args):
        c, G = self.get_1D_center_generator()
        
        # Store data for continuous plotting
        if c_history is not None and G_history is not None:
            if update:
                c_history.append(c)
                G_history.append(G)
            
            steps = np.arange(0, len(c_history))

            min_vals = np.array(c_history) - np.array(G_history)
            max_vals = np.array(c_history) + np.array(G_history)

            min_vals = np.array(min_vals).flatten()
            max_vals = np.array(max_vals).flatten()

        else:
            steps = np.array([0])
            min_vals = np.array([c - G]).flatten()
            max_vals = np.array([c + G]).flatten()

        # Clear previous plots and redraw (for real-time updates)
        if ax is None:
            plt.fill_between(steps, min_vals, max_vals, color=color, alpha=alpha, label=label, edgecolor='black')
            if center:
                plt.plot(steps, c_history, 'ko-')

            # Labels and limits
            plt.xlabel("Step / Iteration")
            plt.ylabel("Interval Range")
            plt.title("1D Zonotope Interval over Steps")
        
        else:
            ax.fill_between(steps, min_vals, max_vals, color=color, alpha=alpha, label=label, edgecolor='black')
            if center:
                ax.plot(steps, np.array(c_history).flatten(), 'ko-')  # Plot center line

            # Labels and limits
            ax.set_xlabel("Step / Iteration")
            ax.set_ylabel("Interval Range")
            ax.set_title("1D Zonotope Interval over Steps")

        return ax
            
    def rand_point(self):
        """
        Random point generation within the zonotope.

        Returns:
            np.array: Random point within the zonotope.
        """
        G = self.generators()
        factors = -1 + 2 * np.random.random((1, G.shape[1]))
        p = self.center() + np.reshape(np.sum(factors * G, axis=1), (G.shape[0], 1))
        
        return p
        
    def quad_map(self, Q):
        """
        Quadratic mapping of the zonotope. 

        Args:
            Q (np.array): Quadratic matrix.

        Returns:
            object: Zonotope object resulting from the quadratic mapping.
        """
        Z_mat = self.Z 
        dimQ = len(Q)
        gens = len(Z_mat[0, :]) - 1
        C = np.zeros((dimQ, 1))
        G = np.zeros((dimQ, int(0.5 * ((np.power(gens, 2)) + gens) + gens)))
        
        Qnonempty = np.zeros((dimQ, 1))

        for i in range(dimQ):
            Qnonempty[i] = np.any(Q[i].reshape((-1, 1)))

            if(Qnonempty[i]):
                QuadMat = np.dot(Z_mat.T, np.dot(Q[i], Z_mat))

                G[i, 0 : gens - 1] = 0.5 * np.diag(QuadMat[1 : gens, 1 : gens])

                C[i, 0] = QuadMat[0, 0] + np.sum(G[i, 0 : gens - 1])

                quadMatoffdiag = QuadMat + QuadMat.T

                quadMatoffdiag = quadMatoffdiag.flatten()

                kInd = np.tril(np.ones((gens + 1, gens + 1)), -1) 
                #print(kInd)
                G[i, gens :] = quadMatoffdiag[kInd.flatten() == 1]

        if(np.sum(Qnonempty) <= 1):
            Zquad = Zonotope(C, np.sum(np.abs(G),1))
        else:
            Zquad = Zonotope(C, self.nonzero_filter(G))

        return Zquad

    def is_intersecting(self, obj, *args):
        """
        Check if the zonotope is intersecting with another object.

        Args:
            obj (object): Object to check intersection with.
            *args: Additional arguments for the intersection check. If no arguments are provided, the default intersection check 
            is used. If one argument is provided, it is the transformation matrix for the intersection check.

        Returns:
            bool: True if the zonotope is intersecting with the object, False otherwise.
        """
        if(isinstance(obj, Zonotope)):
            if(len(args) == 0):
                return self.check_zono_intersection(obj)
            else:
                return self.check_zono_intersection(obj, args)
        elif(isinstance(obj, mptPolytope.mptPolytope)):
            return obj.is_intersecting(self, "exact")

    def check_zono_intersection(self, obj):
        """
        Check if the zonotope is intersecting with another zonotope. 

        Args:
            obj (object): Zonotope object to check intersection with.
        
        Returns:
            bool: True if the zonotope is intersecting with the object, False otherwise.
            delta_center (np.array): Center of the zonotope.
        """
        # Define the center and generators of the zonotope object
        c_1 = self.center()
        G_1 = self.generators()

        # Define the center and generators of the object
        c_2 = obj.center()
        G_2 = obj.generators()

        # Intersect the two zonotopes
        c, G, A, b = self.intersect_zonotopes(obj)

        # Make the linear program for the constrained zonotope empty check
        LP_f,LP_A_ineq,LP_b_ineq,LP_A_eq,LP_b_eq = self.make_con_zono_empty_check_LP(A, b)

        def cost(x, *args):
            """
            Cost function for the linear program. 

            Args:
                x (np.array): Input vector.
                *args: Additional arguments for the cost function.
            """
            A = args[0] 
            result = np.dot(x.reshape((1, len(x))), A)[0]
            return result

        # Define the equality and inequality constraints for the linear program
        eq_con = LinearConstraint(LP_A_eq, LP_b_eq.flatten(), LP_b_eq.flatten())
        ineq_con = LinearConstraint(LP_A_ineq, np.array([np.NINF] * len(LP_b_ineq)), LP_b_ineq.flatten())
        cons = [eq_con, ineq_con]
        
        # Solve the linear program using the high-ipm method
        res = linprog(LP_f, A_ub = LP_A_ineq, b_ub=LP_b_ineq.flatten(), \
                    A_eq=LP_A_eq, b_eq=LP_b_eq.flatten(), bounds=(None,None), \
                    options = {'maxiter':50}, method = 'highs-ipm')

        # Check if the zonotope is empty
        z_opt = res["x"]
        print("z_opt: ", z_opt)
        lm = res["ineqlin"]["marginals"]
        nu = res["eqlin"]["marginals"]
        if(z_opt[-1] > 1):
            return False, 0
        else:
            # Compute the Lagrange multipliers
            M_Q = np.zeros((LP_A_ineq.shape[1], LP_A_ineq.shape[1]))
            M_GT = LP_A_ineq.T
            M_AT = LP_A_eq.T 
            M_DlmG = np.dot(np.diag(lm), LP_A_ineq) 
            M_DGzh = np.diag((np.dot(LP_A_ineq, z_opt) - LP_b_ineq.flatten()))
            M_A = LP_A_eq

            # Compute the Jacobian matrix for the linear program
            row_1 = np.hstack((M_Q, M_GT, M_AT))
            row_2 = np.hstack((M_DlmG, M_DGzh, np.zeros((M_DGzh.shape[0], M_AT.shape[1]))))
            row_3 = np.hstack((M_A, np.zeros((M_A.shape[0], M_DGzh.shape[1] + M_AT.shape[1]))))

            # Compute the left-hand side and right-hand side of the linear program
            LHS = np.vstack((row_1,
                                row_2,
                                row_3))
            db = np.eye(LP_b_eq.shape[0])
            RHS = np.vstack((np.zeros((LHS.shape[0] - db.shape[0], db.shape[1])), db))

            # Compute the Jacobian matrix for the linear program
            J = np.dot(np.linalg.pinv(LHS), RHS) 
            dz_opt_d_c_2 = J[: len(z_opt), :]
            con = 1 - z_opt[-1] * z_opt[-1]
            d_con = -2 * z_opt[-1] * dz_opt_d_c_2[-1, :]

            # Compute the center of the zonotope
            delta_center = np.linalg.pinv(d_con.reshape((1, -1))) * con
            print("delta_center: ", delta_center)

            print(G)

            return True, delta_center

    def intersect_zonotopes(self, obj):
        """
        Intersect two zonotopes. 

        Args:
            obj (object): Zonotope object to intersect with.
        
        Returns:
            np.array: Center of the zonotope.
            np.array: Generator matrix of the zonotope.
            np.array: Inequality matrix.
            np.array: Inequality vector.
        """
        # Define the center and generators of the zonotope object
        c_1 = self.center()
        G_1 = self.generators()

        # Define the center and generators of the object
        c_2 = obj.center()
        G_2 = obj.generators()

        # Define the dimensions of the zonotope
        d = c_1.shape[0]
        n = G_2.shape[1]

        # Define the cost function, inequality matrix, and inequality vector
        G = np.hstack((G_1, np.zeros((d, n))))
        A = np.hstack((G_1, -1 * G_2))
        b = c_2 - c_1

        return c_1, G, A, b

    def make_con_zono_empty_check_LP(self, A, b):
        """
        Make the linear program for the constrained zonotope empty check.

        Args:
            A (np.array): Constraint matrix.
            b (np.array): Constraint vector.

        Returns:
            np.array: Cost function.
            np.array: Inequality matrix.
            np.array: Inequality vector.
            np.array: Equality matrix.
            np.array: Equality vector.
        """
        d = A.shape[1]
        f_cost = np.vstack((np.zeros((d, 1)), np.ones((1, 1))))
        A_ineq = np.vstack((np.hstack((-1 * np.eye(d), -1 * np.ones((d, 1)))), np.hstack((np.eye(d), -1 * np.ones((d, 1))))))
        b_ineq = np.zeros((2 * d, 1))

        A_eq = np.hstack((A, np.zeros((len(A), 1))))
        b_eq = b 

        return f_cost, A_ineq, b_ineq, A_eq, b_eq

    def conv_hull(self):
        """
        Compute the convex hull of the zonotope.

        Returns:
            np.array: Convex hull of the zonotope.
        """
        V = self.polygon()
        hull = ConvexHull(V.T)

        return hull

    def rotate(self, angle):
        """
        Rotate a zonotope based on the previous yaw angle of the vehicle.

        Args:
            zonotope (object): Zonotope to rotate. Dimensions: [X, Y, yaw, v].

        Returns:
            zonotope (object): Rotated zonotope. Dimensions: [X, Y, yaw, v].
        """
        G = self.generators()

        # Rotation matrix
        R_rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # Adjust the generators for rotation
        G_rotated = R_rotation @ G[:2]  # Apply rotation to X-Y plane
        G[:2] = G_rotated

        # Update the zonotope with the rotated generators
        zonotope = Zonotope(self.center(), G)

        return zonotope
    
    def convert2sin_cos(self, lr=1.549):
        """
        Converts a 4D zonotope [x, y, yaw, vx] into a 6D zonotope [x, y, cos(yaw), sin(yaw), vx, w]
        where w = vx / lr.

        Parameters:
            lr (float): The characteristic length (rear axle to center of gravity) of the vehicle.

        Returns:
            Zonotope: A new 6D zonotope in the transformed space.
        """
        # Extract center and generator matrix
        c = self.center().flatten()  # Shape: (4,)
        G = self.generators()  # Shape: (4, ng)

        # Extract yaw and vx indices
        yaw_idx = 2
        vx_idx = 3

        # Compute cos(yaw) and sin(yaw) at the center
        cos_yaw_c = np.cos(c[yaw_idx])
        sin_yaw_c = np.sin(c[yaw_idx])
        w_c = c[vx_idx] / lr

        # Compute new generators for cos(yaw) and sin(yaw) using first-order Taylor expansion
        cos_yaw_G = -sin_yaw_c * G[yaw_idx, :]
        sin_yaw_G = cos_yaw_c * G[yaw_idx, :]
        w_G = (1 / lr) * G[vx_idx, :]

        # Construct the new center
        c_new = np.array([c[0], c[1], cos_yaw_c, sin_yaw_c, c[vx_idx], w_c]).reshape(-1, 1)

        # Construct the new generator matrix
        G_new = np.vstack([
            G[0, :],  # x generators
            G[1, :],  # y generators
            cos_yaw_G,  # cos(yaw) generators
            sin_yaw_G,  # sin(yaw) generators
            G[vx_idx, :],  # vx generators
            w_G  # w generators
        ])

        # Return the transformed zonotope
        return Zonotope(c_new, G_new)
